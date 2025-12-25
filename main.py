from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import requests
import json
from typing import Optional, Dict, Any, List
import os
from datetime import datetime, timedelta
import urllib.parse
from groq import Groq
from rapidfuzz import fuzz
import io
import csv
from collections import defaultdict
from dotenv import load_dotenv

# Initialize FastAPI app
app = FastAPI(title="Zoho Books ERP Chatbot", version="2.0.0")

# Templates for HTML rendering
templates = Jinja2Templates(directory="templates")


load_dotenv()

# Configuration - Replace hardcoded values
ZOHO_CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
ZOHO_CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
ZOHO_ORGANIZATION_ID = os.getenv("ZOHO_ORGANIZATION_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:8000/auth/callback")

# Global variables to store tokens and user info
access_token = None
refresh_token = None
current_user_info = None
current_org_info = None

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# ==================== PHASE 1: CACHING LAYER ====================
class ZohoCache:
    """In-memory cache with TTL for API responses"""
    def __init__(self, ttl_seconds=300):  # 5 minutes default
        self.cache: Dict[str, tuple[Any, datetime]] = {}
        self.ttl = timedelta(seconds=ttl_seconds)
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached data if not expired"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Store data in cache with current timestamp"""
        self.cache[key] = (value, datetime.now())
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
    
    def invalidate(self, pattern: str):
        """Invalidate cache keys matching pattern"""
        keys_to_delete = [k for k in self.cache.keys() if pattern in k]
        for key in keys_to_delete:
            del self.cache[key]

# Initialize cache
zoho_cache = ZohoCache(ttl_seconds=300)

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    page: Optional[int] = 1

class PaginationInfo(BaseModel):
    current_page: int
    total_pages: int
    total_items: int
    items_per_page: int
    has_next: bool
    has_prev: bool

class ChatResponse(BaseModel):
    response: str
    data: Optional[dict] = None
    pagination: Optional[PaginationInfo] = None
    query_context: Optional[dict] = None

class DashboardStats(BaseModel):
    total_revenue: float
    outstanding_amount: float
    overdue_amount: float
    total_customers: int
    total_invoices: int
    total_expenses: float
    paid_invoices: int
    unpaid_invoices: int

# OAuth Flow Functions
def get_auth_url():
    """Generate Zoho OAuth authorization URL"""
    params = {
        'scope': 'ZohoBooks.fullaccess.all',
        'client_id': ZOHO_CLIENT_ID,
        'response_type': 'code',
        'access_type': 'offline',
        'redirect_uri': REDIRECT_URI
    }
    
    base_url = "https://accounts.zoho.com/oauth/v2/auth"
    return f"{base_url}?{urllib.parse.urlencode(params)}"

def exchange_code_for_tokens(auth_code: str):
    """Exchange authorization code for access and refresh tokens"""
    global access_token, refresh_token
    
    token_url = "https://accounts.zoho.com/oauth/v2/token"
    data = {
        'code': auth_code,
        'client_id': ZOHO_CLIENT_ID,
        'client_secret': ZOHO_CLIENT_SECRET,
        'redirect_uri': REDIRECT_URI,
        'grant_type': 'authorization_code'
    }
    
    response = requests.post(token_url, data=data)
    if response.status_code == 200:
        token_data = response.json()
        access_token = token_data.get('access_token')
        refresh_token = token_data.get('refresh_token')
        zoho_cache.clear()  # Clear cache on new auth
        
        # Fetch user and org info after successful auth
        fetch_user_and_org_info()
        
        return True
    return False

def refresh_access_token():
    """Refresh access token using refresh token"""
    global access_token
    
    if not refresh_token:
        return False
        
    token_url = "https://accounts.zoho.com/oauth/v2/token"
    data = {
        'refresh_token': refresh_token,
        'client_id': ZOHO_CLIENT_ID,
        'client_secret': ZOHO_CLIENT_SECRET,
        'grant_type': 'refresh_token'
    }
    
    response = requests.post(token_url, data=data)
    if response.status_code == 200:
        token_data = response.json()
        access_token = token_data.get('access_token')
        zoho_cache.clear()  # Clear cache on token refresh
        return True
    return False

# ==================== USER & ORGANIZATION INFO ====================
def fetch_user_and_org_info():
    """Fetch current user and organization information from Zoho"""
    global current_user_info, current_org_info
    
    try:
        # Get organization info
        org_data = make_zoho_request(f'organizations/{ZOHO_ORGANIZATION_ID}')
        if org_data and 'organization' in org_data:
            current_org_info = org_data['organization']
        
        # Get user info (from organization users endpoint)
        users_data = make_zoho_request('users')
        if users_data and 'users' in users_data:
            # Get the first admin/owner user
            for user in users_data['users']:
                if user.get('role_name') in ['admin', 'owner', 'Administrator']:
                    current_user_info = user
                    break
            # If no admin found, just use first user
            if not current_user_info and users_data['users']:
                current_user_info = users_data['users'][0]
                
    except Exception as e:
        print(f"Error fetching user/org info: {e}")
        current_user_info = None
        current_org_info = None

def get_user_info():
    """Get cached user and organization info"""
    global current_user_info, current_org_info
    
    # If not cached, fetch it
    if not current_user_info or not current_org_info:
        fetch_user_and_org_info()
    
    return {
        'user': current_user_info or {
            'name': 'User',
            'email': '',
            'role_name': ''
        },
        'organization': current_org_info or {
            'name': 'Organization',
            'currency_code': 'SAR'
        }
    }

# ==================== IMPROVED ERROR HANDLING ====================
class ZohoAPIError(Exception):
    """Custom exception for Zoho API errors"""
    def __init__(self, message: str, status_code: int = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

def make_zoho_request(endpoint: str, params: dict = None, use_cache: bool = True):
    """Make authenticated request to Zoho Books API with caching and improved error handling"""
    global access_token
    
    if not access_token:
        raise ZohoAPIError("Not authenticated. Please connect to Zoho Books first.", 401)
    
    # Generate cache key
    cache_key = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
    
    # Check cache first
    if use_cache:
        cached_data = zoho_cache.get(cache_key)
        if cached_data is not None:
            return cached_data
    
    headers = {
        'Authorization': f'Zoho-oauthtoken {access_token}',
        'Content-Type': 'application/json'
    }
    
    base_url = "https://www.zohoapis.com/books/v3"
    url = f"{base_url}/{endpoint}"
    
    if params is None:
        params = {}
    
    # Only add organization_id if not in the endpoint itself
    if 'organizations/' not in endpoint and 'users' not in endpoint:
        params['organization_id'] = ZOHO_ORGANIZATION_ID
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        # Handle 401 - Try to refresh token
        if response.status_code == 401:
            if refresh_access_token():
                headers['Authorization'] = f'Zoho-oauthtoken {access_token}'
                response = requests.get(url, headers=headers, params=params, timeout=30)
            else:
                raise ZohoAPIError("Authentication failed. Please reconnect to Zoho Books.", 401)
        
        # Handle 404
        if response.status_code == 404:
            raise ZohoAPIError(f"Resource not found: {endpoint}", 404)
        
        # Handle 429 - Rate limit
        if response.status_code == 429:
            raise ZohoAPIError("Rate limit exceeded. Please try again in a moment.", 429)
        
        # Handle other errors
        if response.status_code != 200:
            error_msg = f"Zoho API error: {response.status_code}"
            try:
                error_data = response.json()
                if 'message' in error_data:
                    error_msg = error_data['message']
            except:
                pass
            raise ZohoAPIError(error_msg, response.status_code)
        
        data = response.json()
        
        # Cache successful response
        if use_cache:
            zoho_cache.set(cache_key, data)
        
        return data
            
    except requests.Timeout:
        raise ZohoAPIError("Request timed out. Please try again.", 408)
    except requests.ConnectionError:
        raise ZohoAPIError("Connection error. Please check your internet connection.", 503)
    except ZohoAPIError:
        raise
    except Exception as e:
        raise ZohoAPIError(f"Unexpected error: {str(e)}", 500)

# ==================== PHASE 1: FUZZY CUSTOMER MATCHING ====================
def filter_by_customer_fuzzy(items: list, customer_name: str, name_field: str = 'customer_name', threshold: int = 80) -> list:
    """Filter items by customer name using fuzzy matching"""
    if not customer_name or not items:
        return items
    
    customer_name_lower = customer_name.lower()
    results = []
    
    for item in items:
        item_customer = item.get(name_field, '')
        if not item_customer:
            continue
        
        # Calculate fuzzy match score
        score = fuzz.ratio(customer_name_lower, item_customer.lower())
        
        # Also check partial ratio for better matching
        partial_score = fuzz.partial_ratio(customer_name_lower, item_customer.lower())
        
        # Use the higher score
        best_score = max(score, partial_score)
        
        if best_score >= threshold:
            results.append((item, best_score))
    
    # Sort by match score (highest first) and return items
    results.sort(key=lambda x: x[1], reverse=True)
    return [item for item, score in results]

# ==================== ZOHO API FUNCTIONS ====================
def get_invoices(status: str = None, customer_name: str = None, limit: int = 25, page: int = 1):
    """Get invoices from Zoho Books with fuzzy customer matching"""
    fetch_limit = 200 if customer_name else min(limit, 200)
    
    params = {
        'per_page': fetch_limit,
        'page': 1
    }
    if status:
        params['status'] = status
    
    data = make_zoho_request('invoices', params)
    
    # Apply fuzzy customer filtering
    if data and customer_name and 'invoices' in data:
        data['invoices'] = filter_by_customer_fuzzy(data['invoices'], customer_name, 'customer_name')
        if 'page_context' in data:
            data['page_context']['total'] = len(data['invoices'])
    
    return data

def get_quotes(status: str = None, customer_name: str = None, limit: int = 25, page: int = 1):
    """Get quotes/estimates from Zoho Books"""
    fetch_limit = 200 if customer_name else min(limit, 200)
    
    params = {
        'per_page': fetch_limit,
        'page': 1
    }
    if status:
        params['status'] = status
    
    data = make_zoho_request('estimates', params)
    
    if data and customer_name and 'estimates' in data:
        data['estimates'] = filter_by_customer_fuzzy(data['estimates'], customer_name, 'customer_name')
        if 'page_context' in data:
            data['page_context']['total'] = len(data['estimates'])
    
    return data

def get_salesorders(status: str = None, customer_name: str = None, limit: int = 25, page: int = 1):
    """Get sales orders from Zoho Books"""
    fetch_limit = 200 if customer_name else min(limit, 200)
    
    params = {
        'per_page': fetch_limit,
        'page': 1
    }
    if status:
        params['status'] = status
    
    data = make_zoho_request('salesorders', params)
    
    if data and customer_name and 'salesorders' in data:
        data['salesorders'] = filter_by_customer_fuzzy(data['salesorders'], customer_name, 'customer_name')
        if 'page_context' in data:
            data['page_context']['total'] = len(data['salesorders'])
    
    return data

# ==================== PHASE 1: BILLS & PAYMENTS TRACKING ====================
def get_bills(status: str = None, vendor_name: str = None, limit: int = 25, page: int = 1):
    """Get bills from Zoho Books"""
    fetch_limit = 200 if vendor_name else min(limit, 200)
    
    params = {
        'per_page': fetch_limit,
        'page': 1
    }
    if status:
        params['status'] = status
    
    data = make_zoho_request('bills', params)
    
    if data and vendor_name and 'bills' in data:
        data['bills'] = filter_by_customer_fuzzy(data['bills'], vendor_name, 'vendor_name')
        if 'page_context' in data:
            data['page_context']['total'] = len(data['bills'])
    
    return data

def get_payments(customer_name: str = None, limit: int = 25, page: int = 1):
    """Get customer payments from Zoho Books"""
    params = {
        'per_page': min(limit, 200),
        'page': page
    }
    
    data = make_zoho_request('customerpayments', params)
    
    if data and customer_name and 'customerpayments' in data:
        data['customerpayments'] = filter_by_customer_fuzzy(
            data['customerpayments'], 
            customer_name, 
            'customer_name'
        )
        if 'page_context' in data:
            data['page_context']['total'] = len(data['customerpayments'])
    
    return data

def get_vendor_credits(limit: int = 25, page: int = 1):
    """Get vendor credits from Zoho Books"""
    params = {
        'per_page': min(limit, 200),
        'page': page
    }
    return make_zoho_request('vendorcredits', params)

# ==================== PHASE 2: RECURRING INVOICES & CREDIT NOTES ====================
def get_recurring_invoices(status: str = None, customer_name: str = None, limit: int = 25):
    """Get recurring invoices from Zoho Books"""
    params = {
        'per_page': min(limit, 200)
    }
    if status:
        params['status'] = status
    
    data = make_zoho_request('recurringinvoices', params)
    
    if data and customer_name and 'recurring_invoices' in data:
        data['recurring_invoices'] = filter_by_customer_fuzzy(
            data['recurring_invoices'], 
            customer_name, 
            'customer_name'
        )
    
    return data

def get_credit_notes(customer_name: str = None, limit: int = 25):
    """Get credit notes from Zoho Books"""
    params = {
        'per_page': min(limit, 200)
    }
    
    data = make_zoho_request('creditnotes', params)
    
    if data and customer_name and 'creditnotes' in data:
        data['creditnotes'] = filter_by_customer_fuzzy(
            data['creditnotes'], 
            customer_name, 
            'customer_name'
        )
    
    return data

# ==================== EXISTING FUNCTIONS ====================
def search_by_document_number(doc_number: str, doc_type: str):
    """Search for a specific document by number"""
    endpoint_map = {
        'quote': 'estimates',
        'invoice': 'invoices',
        'salesorder': 'salesorders'
    }
    
    endpoint = endpoint_map.get(doc_type, 'invoices')
    
    params = {
        'per_page': 200,
        'page': 1
    }
    
    data = make_zoho_request(endpoint, params)
    
    if not data:
        return None
    
    key_map = {
        'estimates': 'estimates',
        'invoices': 'invoices', 
        'salesorders': 'salesorders'
    }
    
    items_key = key_map.get(endpoint, 'invoices')
    items = data.get(items_key, [])
    
    number_field_map = {
        'estimates': 'estimate_number',
        'invoices': 'invoice_number',
        'salesorders': 'salesorder_number'
    }
    
    number_field = number_field_map.get(endpoint, 'invoice_number')
    
    for item in items:
        if item.get(number_field, '').upper() == doc_number.upper():
            return {items_key: [item]}
    
    return None

def get_customers(limit: int = 25, page: int = 1):
    """Get customers from Zoho Books"""
    params = {
        'per_page': min(limit, 200),
        'page': page
    }
    return make_zoho_request('contacts', params)

def get_expenses(limit: int = 25, page: int = 1):
    """Get expenses from Zoho Books"""
    params = {
        'per_page': min(limit, 200),
        'page': page
    }
    return make_zoho_request('expenses', params)

def get_reports_revenue():
    """Get revenue reports"""
    return make_zoho_request('reports/profitandloss')

# ==================== PHASE 2: AGING REPORTS ====================
def calculate_aging_buckets(invoices: List[dict]) -> Dict[str, Any]:
    """Calculate aging analysis for invoices (30/60/90+ days)"""
    today = datetime.now().date()
    
    aging = {
        'current': {'count': 0, 'amount': 0, 'invoices': []},
        '1-30': {'count': 0, 'amount': 0, 'invoices': []},
        '31-60': {'count': 0, 'amount': 0, 'invoices': []},
        '61-90': {'count': 0, 'amount': 0, 'invoices': []},
        '90+': {'count': 0, 'amount': 0, 'invoices': []}
    }
    
    for invoice in invoices:
        # Only consider unpaid/partially paid invoices
        balance = float(invoice.get('balance', 0))
        if balance <= 0:
            continue
        
        due_date_str = invoice.get('due_date', '')
        if not due_date_str:
            continue
        
        try:
            due_date = datetime.strptime(due_date_str, '%Y-%m-%d').date()
            days_overdue = (today - due_date).days
            
            invoice_summary = {
                'number': invoice.get('invoice_number', 'N/A'),
                'customer': invoice.get('customer_name', 'N/A'),
                'amount': balance,
                'due_date': due_date_str,
                'days_overdue': days_overdue
            }
            
            if days_overdue <= 0:
                bucket = 'current'
            elif days_overdue <= 30:
                bucket = '1-30'
            elif days_overdue <= 60:
                bucket = '31-60'
            elif days_overdue <= 90:
                bucket = '61-90'
            else:
                bucket = '90+'
            
            aging[bucket]['count'] += 1
            aging[bucket]['amount'] += balance
            aging[bucket]['invoices'].append(invoice_summary)
            
        except ValueError:
            continue
    
    return aging

def get_aging_report():
    """Get aging report for receivables"""
    try:
        # Get all unpaid and partially paid invoices
        invoices_data = get_invoices(limit=200)
        
        if not invoices_data or 'invoices' not in invoices_data:
            return None
        
        invoices = invoices_data['invoices']
        
        # Filter for unpaid/partially paid
        unpaid_invoices = [inv for inv in invoices if float(inv.get('balance', 0)) > 0]
        
        # Calculate aging buckets
        aging = calculate_aging_buckets(unpaid_invoices)
        
        # Calculate totals
        total_outstanding = sum(bucket['amount'] for bucket in aging.values())
        
        return {
            'aging_buckets': aging,
            'total_outstanding': total_outstanding,
            'total_invoices': sum(bucket['count'] for bucket in aging.values()),
            'generated_date': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Aging report error: {e}")
        return None

# ==================== PHASE 2: PAYMENT REMINDERS ====================
def get_overdue_invoices():
    """Get list of overdue invoices that need payment reminders"""
    try:
        invoices_data = get_invoices(status='overdue', limit=200)
        
        if not invoices_data or 'invoices' not in invoices_data:
            return []
        
        overdue = []
        today = datetime.now().date()
        
        for invoice in invoices_data['invoices']:
            balance = float(invoice.get('balance', 0))
            if balance <= 0:
                continue
            
            due_date_str = invoice.get('due_date', '')
            if due_date_str:
                try:
                    due_date = datetime.strptime(due_date_str, '%Y-%m-%d').date()
                    days_overdue = (today - due_date).days
                    
                    overdue.append({
                        'invoice_number': invoice.get('invoice_number', 'N/A'),
                        'customer_name': invoice.get('customer_name', 'N/A'),
                        'balance': balance,
                        'due_date': due_date_str,
                        'days_overdue': days_overdue,
                        'total': float(invoice.get('total', 0))
                    })
                except ValueError:
                    continue
        
        # Sort by days overdue (most overdue first)
        overdue.sort(key=lambda x: x['days_overdue'], reverse=True)
        
        return overdue
        
    except Exception as e:
        print(f"Overdue invoices error: {e}")
        return []

# ==================== PHASE 2: DASHBOARD STATS ====================
def get_dashboard_stats() -> Dict[str, Any]:
    """Calculate comprehensive dashboard statistics"""
    try:
        # Fetch data (will use cache if available)
        invoices_data = get_invoices(limit=200)
        expenses_data = get_expenses(limit=200)
        customers_data = get_customers(limit=200)
        
        stats = {
            'total_revenue': 0,
            'outstanding_amount': 0,
            'overdue_amount': 0,
            'total_customers': 0,
            'total_invoices': 0,
            'total_expenses': 0,
            'paid_invoices': 0,
            'unpaid_invoices': 0,
            'monthly_revenue': defaultdict(float),
            'invoice_status_breakdown': defaultdict(int),
            'top_customers': [],
            'recent_invoices': [],
            'expense_trend': []
        }
        
        # Process invoices
        if invoices_data and 'invoices' in invoices_data:
            invoices = invoices_data['invoices']
            stats['total_invoices'] = len(invoices)
            
            customer_revenue = defaultdict(float)
            
            for invoice in invoices:
                total = float(invoice.get('total', 0))
                balance = float(invoice.get('balance', 0))
                status = invoice.get('status', 'unknown')
                customer = invoice.get('customer_name', 'Unknown')
                
                stats['total_revenue'] += total
                stats['outstanding_amount'] += balance
                
                # Count status
                stats['invoice_status_breakdown'][status] += 1
                
                if status == 'paid':
                    stats['paid_invoices'] += 1
                elif status in ['sent', 'viewed', 'overdue']:
                    stats['unpaid_invoices'] += 1
                    
                if status == 'overdue':
                    stats['overdue_amount'] += balance
                
                # Monthly revenue
                date_str = invoice.get('date', '')
                if date_str:
                    try:
                        date = datetime.strptime(date_str, '%Y-%m-%d')
                        month_key = date.strftime('%Y-%m')
                        stats['monthly_revenue'][month_key] += total
                    except ValueError:
                        pass
                
                # Customer revenue
                customer_revenue[customer] += total
            
            # Top customers
            top_customers = sorted(customer_revenue.items(), key=lambda x: x[1], reverse=True)[:5]
            stats['top_customers'] = [
                {'customer': cust, 'revenue': rev} for cust, rev in top_customers
            ]
            
            # Recent invoices
            stats['recent_invoices'] = invoices[:5]
        
        # Process expenses
        if expenses_data and 'expenses' in expenses_data:
            expenses = expenses_data['expenses']
            stats['total_expenses'] = sum(float(exp.get('total', 0)) for exp in expenses)
        
        # Process customers
        if customers_data and 'contacts' in customers_data:
            stats['total_customers'] = len(customers_data['contacts'])
        
        # Calculate profit
        stats['net_profit'] = stats['total_revenue'] - stats['total_expenses']
        
        return stats
        
    except Exception as e:
        print(f"Dashboard stats error: {e}")
        return None

# ==================== PHASE 2: CHART DATA ====================
def get_chart_data():
    """Generate chart-ready data for visualization"""
    try:
        stats = get_dashboard_stats()
        
        if not stats:
            return None
        
        # Monthly revenue trend (last 6 months)
        monthly_data = dict(sorted(stats['monthly_revenue'].items())[-6:])
        revenue_chart = {
            'labels': list(monthly_data.keys()),
            'data': list(monthly_data.values())
        }
        
        # Invoice status distribution
        status_chart = {
            'labels': list(stats['invoice_status_breakdown'].keys()),
            'data': list(stats['invoice_status_breakdown'].values())
        }
        
        # Top customers
        top_customers_chart = {
            'labels': [item['customer'] for item in stats['top_customers']],
            'data': [item['revenue'] for item in stats['top_customers']]
        }
        
        return {
            'revenue_trend': revenue_chart,
            'invoice_status': status_chart,
            'top_customers': top_customers_chart
        }
        
    except Exception as e:
        print(f"Chart data error: {e}")
        return None

# ==================== PHASE 1: EXPORT FUNCTIONALITY ====================
def export_to_csv(data: List[dict], data_type: str) -> io.StringIO:
    """Export data to CSV format"""
    output = io.StringIO()
    
    if not data:
        return output
    
    # Define columns based on data type
    columns_map = {
        'invoices': ['invoice_number', 'customer_name', 'date', 'due_date', 'total', 'balance', 'status'],
        'expenses': ['date', 'account_name', 'description', 'total', 'vendor_name', 'reference_number'],
        'customers': ['contact_name', 'company_name', 'email', 'phone', 'outstanding_receivable_amount'],
        'quotes': ['estimate_number', 'customer_name', 'date', 'total', 'status', 'reference_number'],
        'salesorders': ['salesorder_number', 'customer_name', 'date', 'total', 'status', 'shipment_date'],
        'bills': ['bill_number', 'vendor_name', 'date', 'due_date', 'total', 'balance', 'status'],
        'payments': ['payment_number', 'customer_name', 'date', 'amount', 'payment_mode', 'reference_number']
    }
    
    columns = columns_map.get(data_type, list(data[0].keys()) if data else [])
    
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    
    for item in data:
        writer.writerow(item)
    
    output.seek(0)
    return output

# ==================== AI PROCESSING ====================
def get_limit_from_scope(scope: str, requested_count: int = None):
    """Convert scope to actual item limits"""
    scope_map = {
        'small': {'fetch': 10, 'display': 10},
        'medium': {'fetch': 25, 'display': 25},
        'xlarge': {'fetch': 50, 'display': 50}
    }
    
    limits = scope_map.get(scope, scope_map['medium'])
    
    if requested_count:
        limits['fetch'] = min(requested_count, 200)
        limits['display'] = min(requested_count, 200)
    
    return limits

def calculate_pagination(data: dict, page: int, per_page: int):
    """Calculate pagination information"""
    page_context = data.get('page_context', {})
    
    if 'invoices' in data:
        items = data.get('invoices', [])
        total_items = page_context.get('total', len(items))
    elif 'estimates' in data:
        items = data.get('estimates', [])
        total_items = page_context.get('total', len(items))
    elif 'salesorders' in data:
        items = data.get('salesorders', [])
        total_items = page_context.get('total', len(items))
    elif 'contacts' in data:
        items = data.get('contacts', [])
        total_items = page_context.get('total', len(items))
    elif 'expenses' in data:
        items = data.get('expenses', [])
        total_items = page_context.get('total', len(items))
    elif 'bills' in data:
        items = data.get('bills', [])
        total_items = page_context.get('total', len(items))
    elif 'customerpayments' in data:
        items = data.get('customerpayments', [])
        total_items = page_context.get('total', len(items))
    else:
        return None
    
    total_pages = (total_items + per_page - 1) // per_page
    
    return PaginationInfo(
        current_page=page,
        total_pages=total_pages,
        total_items=total_items,
        items_per_page=per_page,
        has_next=page < total_pages,
        has_prev=page > 1
    )

def process_query_with_ai(user_query: str):
    """Use Groq AI to understand user intent"""
    
    system_prompt = """You are an AI assistant for a Zoho Books ERP system. Your job is to understand user queries and respond appropriately.

Available actions:
1. "conversation" - for greetings, casual chat, help requests, or general questions
2. "get_invoices" - for queries about invoices, bills, payments
3. "get_invoices_unpaid" - specifically for unpaid/outstanding invoices
4. "get_quotes" - for queries about quotes, quotations, estimates
5. "get_salesorders" - for queries about sales orders, orders
6. "get_customers" - for queries about customers, clients
7. "get_expenses" - for queries about expenses, costs, spending
8. "get_revenue" - for queries about revenue, sales, profit
9. "get_bills" - for queries about vendor bills, payables
10. "get_payments" - for queries about customer payments received
11. "get_recurring" - for queries about recurring invoices, subscriptions
12. "get_credits" - for queries about credit notes, refunds
13. "get_aging" - for queries about aging analysis, overdue analysis, 30/60/90 reports
14. "get_reminders" - for queries about payment reminders, overdue invoices needing follow-up
15. "search_by_number" - when user provides a specific document number

Customer/Company Name Detection:
- If user mentions a specific company/customer name, extract it EXACTLY as written
- Customer filter applies to invoices, quotes, sales orders, bills, payments

Document Number Detection:
- If user provides a number starting with QT-, INV-, SO-, extract it

Detect the scope of data requested:
- "recent", "latest", "last few", "top" = small (10 items)
- "show me", default queries = medium (25 items)
- "all", "complete", "full list", "every", "everything" = xlarge (200 items)

Respond with JSON format:
{
    "action": "action_name",
    "intent": "brief description",
    "scope": "small|medium|xlarge",
    "requested_count": number or null,
    "customer_name": "exact customer name" or null,
    "document_number": "document number" or null,
    "document_type": "quote|invoice|salesorder" or null,
    "response": "friendly response for conversation actions"
}

Examples:
- "show me bills" -> {"action": "get_bills", "intent": "show vendor bills", "scope": "medium"}
- "what payments did we receive?" -> {"action": "get_payments", "intent": "customer payments", "scope": "medium"}
- "aging report" -> {"action": "get_aging", "intent": "aging analysis", "scope": "medium"}
- "overdue invoices needing reminders" -> {"action": "get_reminders", "intent": "payment reminders", "scope": "medium"}
- "recurring invoices" -> {"action": "get_recurring", "intent": "subscription invoices", "scope": "medium"}
- "credit notes" -> {"action": "get_credits", "intent": "customer credits", "scope": "medium"}
"""

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.3,
            max_tokens=400
        )
        
        response_text = completion.choices[0].message.content
        try:
            parsed = json.loads(response_text)
            if 'customer_name' not in parsed:
                parsed['customer_name'] = None
            if 'document_number' not in parsed:
                parsed['document_number'] = None
            if 'document_type' not in parsed:
                parsed['document_type'] = None
            return parsed
        except:
            return {
                "action": "conversation", 
                "intent": "fallback", 
                "scope": "medium",
                "requested_count": None,
                "customer_name": None,
                "document_number": None,
                "document_type": None,
                "response": "I'm here to help you with your Zoho Books data!"
            }
            
    except Exception as e:
        print(f"AI processing error: {e}")
        return {
            "action": "conversation",
            "intent": "error",
            "scope": "medium",
            "requested_count": None,
            "customer_name": None,
            "document_number": None,
            "document_type": None,
            "response": "I'm your Zoho Books assistant. Ask me about invoices, quotes, orders, expenses, bills, payments, or reports!"
        }

def format_response_with_ai(data: dict, intent: str, customer_name: str = None, display_limit: int = 25):
    """Use AI to format the API response into natural language"""
    
    limited_data = limit_data_for_ai(data, display_limit)
    
    showing_all = limited_data.get('showing_count', 0) == limited_data.get('total_count', 0)
    total_items = limited_data.get('total_count', 0)
    
    if total_items > 50:
        return format_data_manually(limited_data, intent, customer_name)
    
    customer_context = f" for {customer_name}" if customer_name else ""
    
    system_prompt = f"""You are an AI assistant that formats business/financial data into clear, executive-friendly responses. 

Create a concise, professional summary. Focus on key insights and numbers.
Use bullet points or numbered lists for clarity.

{"IMPORTANT: The user requested ALL data and we are showing EVERYTHING. Do NOT suggest asking for more data." if showing_all else "IMPORTANT: If showing_count is less than total_count, mention that there are more items available."}

{f"IMPORTANT: This data is filtered for customer: {customer_name}" if customer_name else ""}
"""

    user_prompt = f"Intent: {intent}{customer_context}\n\nData summary: {json.dumps(limited_data, indent=2)}\n\nFormat this into a clear, structured response."

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        print(f"AI formatting error: {e}")
        return format_data_manually(limited_data, intent, customer_name)

def limit_data_for_ai(data: dict, max_items: int = 10):
    """Limit data size for AI processing"""
    if not data:
        return {}
    
    limited = {}
    
    # Handle different data types
    data_handlers = {
        'invoices': lambda inv: {
            'invoice_number': inv.get('invoice_number', 'N/A'),
            'customer_name': inv.get('customer_name', 'N/A'),
            'total': inv.get('total', 0),
            'balance': inv.get('balance', 0),
            'status': inv.get('status', 'N/A'),
            'due_date': inv.get('due_date', 'N/A'),
            'date': inv.get('date', 'N/A')
        },
        'estimates': lambda est: {
            'estimate_number': est.get('estimate_number', 'N/A'),
            'customer_name': est.get('customer_name', 'N/A'),
            'total': est.get('total', 0),
            'status': est.get('status', 'N/A'),
            'date': est.get('date', 'N/A')
        },
        'salesorders': lambda order: {
            'salesorder_number': order.get('salesorder_number', 'N/A'),
            'customer_name': order.get('customer_name', 'N/A'),
            'total': order.get('total', 0),
            'status': order.get('status', 'N/A'),
            'date': order.get('date', 'N/A')
        },
        'contacts': lambda contact: {
            'contact_name': contact.get('contact_name', 'N/A'),
            'company_name': contact.get('company_name', 'N/A'),
            'outstanding_receivable_amount': contact.get('outstanding_receivable_amount', 0)
        },
        'expenses': lambda exp: {
            'date': exp.get('date', 'N/A'),
            'description': exp.get('description', 'N/A'),
            'total': exp.get('total', 0),
            'account_name': exp.get('account_name', 'N/A')
        },
        'bills': lambda bill: {
            'bill_number': bill.get('bill_number', 'N/A'),
            'vendor_name': bill.get('vendor_name', 'N/A'),
            'total': bill.get('total', 0),
            'balance': bill.get('balance', 0),
            'status': bill.get('status', 'N/A'),
            'due_date': bill.get('due_date', 'N/A')
        },
        'customerpayments': lambda pay: {
            'payment_number': pay.get('payment_number', 'N/A'),
            'customer_name': pay.get('customer_name', 'N/A'),
            'amount': pay.get('amount', 0),
            'date': pay.get('date', 'N/A'),
            'payment_mode': pay.get('payment_mode', 'N/A')
        }
    }
    
    for key, handler in data_handlers.items():
        if key in data:
            items = data[key][:max_items]
            limited[key] = [handler(item) for item in items]
            limited['total_count'] = len(data[key])
            limited['showing_count'] = len(items)
            break
    else:
        limited = data
    
    return limited

def format_data_manually(data: dict, intent: str, customer_name: str = None):
    """Manual formatting fallback"""
    
    customer_header = f" for {customer_name}" if customer_name else ""
    
    # Determine data type and format accordingly
    if 'invoices' in data:
        return format_invoices_manual(data, customer_header)
    elif 'bills' in data:
        return format_bills_manual(data, customer_header)
    elif 'customerpayments' in data:
        return format_payments_manual(data, customer_header)
    elif 'quotes' in data or 'estimates' in data:
        return format_quotes_manual(data, customer_header)
    elif 'salesorders' in data:
        return format_salesorders_manual(data, customer_header)
    elif 'contacts' in data:
        return format_customers_manual(data)
    elif 'expenses' in data:
        return format_expenses_manual(data)
    else:
        return f"Found data for: {intent}\n\nSummary: {len(data)} items retrieved successfully."

def format_invoices_manual(data: dict, customer_header: str = ""):
    """Format invoices manually"""
    invoices = data['invoices']
    total_count = data.get('total_count', len(invoices))
    showing_count = data.get('showing_count', len(invoices))
    
    status_groups = defaultdict(lambda: {'count': 0, 'total': 0})
    
    for inv in invoices:
        status = inv.get('status', 'Unknown')
        total = float(inv.get('total', 0))
        status_groups[status]['count'] += 1
        status_groups[status]['total'] += total
    
    response = f"**Invoice Summary{customer_header}**\n\n"
    response += f"• Total Invoices: {total_count}\n"
    response += f"• Showing: {showing_count}\n\n"
    
    response += "**Invoice Status:**\n"
    for status, stats in status_groups.items():
        response += f"• {status}: {stats['count']} invoices, SAR {stats['total']:,.2f}\n"
    
    if showing_count < total_count:
        response += f"\nShowing {showing_count} of {total_count} items."
    
    return response

def format_bills_manual(data: dict, customer_header: str = ""):
    """Format bills manually"""
    bills = data['bills']
    total_count = data.get('total_count', len(bills))
    showing_count = data.get('showing_count', len(bills))
    
    total_amount = sum(float(bill.get('total', 0)) for bill in bills)
    outstanding = sum(float(bill.get('balance', 0)) for bill in bills)
    
    response = f"**Bills Summary{customer_header}**\n\n"
    response += f"• Total Bills: {total_count}\n"
    response += f"• Showing: {showing_count}\n"
    response += f"• Total Amount: SAR {total_amount:,.2f}\n"
    response += f"• Outstanding: SAR {outstanding:,.2f}\n\n"
    
    response += "**Recent Bills:**\n"
    for i, bill in enumerate(bills[:10], 1):
        response += f"{i}. {bill.get('bill_number', 'N/A')} - {bill.get('vendor_name', 'N/A')} - SAR {bill.get('total', 0):,.2f} ({bill.get('status', 'N/A')})\n"
    
    return response

def format_payments_manual(data: dict, customer_header: str = ""):
    """Format payments manually"""
    payments = data['customerpayments']
    total_count = data.get('total_count', len(payments))
    showing_count = data.get('showing_count', len(payments))
    
    total_amount = sum(float(pay.get('amount', 0)) for pay in payments)
    
    response = f"**Payments Summary{customer_header}**\n\n"
    response += f"• Total Payments: {total_count}\n"
    response += f"• Showing: {showing_count}\n"
    response += f"• Total Amount: SAR {total_amount:,.2f}\n\n"
    
    response += "**Recent Payments:**\n"
    for i, pay in enumerate(payments[:10], 1):
        response += f"{i}. {pay.get('payment_number', 'N/A')} - {pay.get('customer_name', 'N/A')} - SAR {pay.get('amount', 0):,.2f} ({pay.get('date', 'N/A')})\n"
    
    return response

def format_quotes_manual(data: dict, customer_header: str = ""):
    """Format quotes manually"""
    quotes = data.get('quotes', data.get('estimates', []))
    total_count = data.get('total_count', len(quotes))
    showing_count = data.get('showing_count', len(quotes))
    
    response = f"**Quote Summary{customer_header}**\n\n"
    response += f"• Showing {showing_count} of {total_count} quotes\n\n"
    
    response += "**Recent quotes:**\n"
    for i, quote in enumerate(quotes[:10], 1):
        response += f"{i}. {quote.get('estimate_number', 'N/A')} - {quote.get('customer_name', 'N/A')} - SAR {quote.get('total', 0):,.2f} ({quote.get('status', 'N/A')})\n"
    
    return response

def format_salesorders_manual(data: dict, customer_header: str = ""):
    """Format sales orders manually"""
    orders = data['salesorders']
    total_count = data.get('total_count', len(orders))
    showing_count = data.get('showing_count', len(orders))
    
    response = f"**Sales Order Summary{customer_header}**\n\n"
    response += f"• Showing {showing_count} of {total_count} orders\n\n"
    
    response += "**Recent orders:**\n"
    for i, order in enumerate(orders[:10], 1):
        response += f"{i}. {order.get('salesorder_number', 'N/A')} - {order.get('customer_name', 'N/A')} - SAR {order.get('total', 0):,.2f} ({order.get('status', 'N/A')})\n"
    
    return response

def format_customers_manual(data: dict):
    """Format customers manually"""
    contacts = data['contacts']
    total_count = data.get('total_count', len(contacts))
    showing_count = data.get('showing_count', len(contacts))
    
    response = f"**Customer Summary**\n\n"
    response += f"• Showing {showing_count} of {total_count} customers\n\n"
    
    response += "**Customers:**\n"
    for i, contact in enumerate(contacts[:10], 1):
        outstanding = contact.get('outstanding_receivable_amount', 0)
        response += f"{i}. {contact.get('contact_name', 'N/A')}"
        if outstanding > 0:
            response += f" (Outstanding: SAR {outstanding:,.2f})"
        response += "\n"
    
    return response

def format_expenses_manual(data: dict):
    """Format expenses manually"""
    expenses = data['expenses']
    total_count = data.get('total_count', len(expenses))
    showing_count = data.get('showing_count', len(expenses))
    total_amount = sum(float(exp.get('total', 0)) for exp in expenses)
    
    response = f"**Expense Summary**\n\n"
    response += f"• Showing {showing_count} of {total_count} expenses\n"
    response += f"• Total amount: SAR {total_amount:,.2f}\n\n"
    
    response += "**Recent expenses:**\n"
    for i, exp in enumerate(expenses[:10], 1):
        response += f"{i}. {exp.get('description', 'N/A')} - SAR {exp.get('total', 0):,.2f} ({exp.get('date', 'N/A')})\n"
    
    return response

# ==================== ROUTES ====================
@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    """Serve the landing page"""
    return templates.TemplateResponse("landing.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Serve the chat interface"""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Serve the dashboard interface"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/auth")
async def start_auth():
    """Start OAuth flow"""
    auth_url = get_auth_url()
    return RedirectResponse(url=auth_url)

@app.get("/auth/callback")
async def auth_callback(code: str = None, error: str = None):
    """Handle OAuth callback"""
    if error:
        return RedirectResponse(url=f"/?error={error}")
    
    if not code:
        return RedirectResponse(url="/?error=no_code")
    
    if exchange_code_for_tokens(code):
        return RedirectResponse(url="/dashboard")
    else:
        return RedirectResponse(url="/?error=auth_failed")

@app.get("/auth/status")
async def auth_status():
    """Check authentication status"""
    return {"authenticated": access_token is not None}

@app.get("/api/user/info")
async def user_info_endpoint():
    """Get current user and organization information"""
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        user_info = get_user_info()
        return user_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== PHASE 2: DASHBOARD API ====================
@app.get("/api/dashboard/stats")
async def dashboard_stats_endpoint():
    """Get dashboard statistics"""
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        stats = get_dashboard_stats()
        if stats is None:
            raise HTTPException(status_code=500, detail="Failed to fetch dashboard stats")
        return stats
    except ZohoAPIError as e:
        raise HTTPException(status_code=e.status_code or 500, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard/charts")
async def dashboard_charts_endpoint():
    """Get chart data for visualization"""
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        charts = get_chart_data()
        if charts is None:
            raise HTTPException(status_code=500, detail="Failed to fetch chart data")
        return charts
    except ZohoAPIError as e:
        raise HTTPException(status_code=e.status_code or 500, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/reports/aging")
async def aging_report_endpoint():
    """Get aging report"""
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        report = get_aging_report()
        if report is None:
            raise HTTPException(status_code=500, detail="Failed to generate aging report")
        return report
    except ZohoAPIError as e:
        raise HTTPException(status_code=e.status_code or 500, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/reports/overdue")
async def overdue_report_endpoint():
    """Get overdue invoices for payment reminders"""
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        overdue = get_overdue_invoices()
        return {
            'overdue_invoices': overdue,
            'total_count': len(overdue),
            'total_amount': sum(inv['balance'] for inv in overdue)
        }
    except ZohoAPIError as e:
        raise HTTPException(status_code=e.status_code or 500, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== PHASE 1: EXPORT ENDPOINTS ====================
@app.get("/api/export/invoices/csv")
async def export_invoices_csv(customer_name: Optional[str] = None, status: Optional[str] = None):
    """Export invoices to CSV"""
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        data = get_invoices(status=status, customer_name=customer_name, limit=200)
        
        if not data or 'invoices' not in data or not data['invoices']:
            raise HTTPException(status_code=404, detail="No invoices found to export")
        
        csv_output = export_to_csv(data['invoices'], 'invoices')
        
        return StreamingResponse(
            iter([csv_output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=invoices_{datetime.now().strftime('%Y%m%d')}.csv"}
        )
    except ZohoAPIError as e:
        raise HTTPException(status_code=e.status_code or 500, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export/expenses/csv")
async def export_expenses_csv():
    """Export expenses to CSV"""
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        data = get_expenses(limit=200)
        
        if not data or 'expenses' not in data or not data['expenses']:
            raise HTTPException(status_code=404, detail="No expenses found to export")
        
        csv_output = export_to_csv(data['expenses'], 'expenses')
        
        return StreamingResponse(
            iter([csv_output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=expenses_{datetime.now().strftime('%Y%m%d')}.csv"}
        )
    except ZohoAPIError as e:
        raise HTTPException(status_code=e.status_code or 500, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export/customers/csv")
async def export_customers_csv():
    """Export customers to CSV"""
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        data = get_customers(limit=200)
        
        if not data or 'contacts' not in data or not data['contacts']:
            raise HTTPException(status_code=404, detail="No customers found to export")
        
        csv_output = export_to_csv(data['contacts'], 'customers')
        
        return StreamingResponse(
            iter([csv_output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=customers_{datetime.now().strftime('%Y%m%d')}.csv"}
        )
    except ZohoAPIError as e:
        raise HTTPException(status_code=e.status_code or 500, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== CHAT ENDPOINT ====================
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    """Main chat endpoint with all new features integrated"""
    if not access_token:
        return ChatResponse(
            response="⚠️ Please authenticate first by connecting to Zoho Books",
            data=None
        )
    
    try:
        ai_intent = process_query_with_ai(message.message)
        action = ai_intent.get("action", "conversation")
        intent = ai_intent.get("intent", "general query")
        scope = ai_intent.get("scope", "medium")
        requested_count = ai_intent.get("requested_count", None)
        customer_name = ai_intent.get("customer_name", None)
        document_number = ai_intent.get("document_number", None)
        document_type = ai_intent.get("document_type", None)
        page = message.page
        
        # Handle conversation actions
        if action == "conversation":
            response_text = ai_intent.get("response", "I'm here to help! Ask me about invoices, quotes, bills, payments, expenses, and more.")
            return ChatResponse(
                response=response_text,
                data=None
            )
        
        # Handle document search
        if action == "search_by_number" and document_number:
            data = search_by_document_number(document_number, document_type or 'invoice')
            
            if data is None or not data:
                return ChatResponse(
                    response=f"❌ Document {document_number} not found. Please check the number and try again.",
                    data=None
                )
            
            formatted_response = format_response_with_ai(data, intent, customer_name, display_limit=1)
            return ChatResponse(
                response=formatted_response,
                data=data
            )
        
        # Handle aging report
        if action == "get_aging":
            report = get_aging_report()
            if report is None:
                return ChatResponse(
                    response="❌ Could not generate aging report. Please try again.",
                    data=None
                )
            
            # Format aging report
            response = "**Accounts Receivable Aging Report**\n\n"
            for bucket, data in report['aging_buckets'].items():
                if data['count'] > 0:
                    response += f"**{bucket} days:**\n"
                    response += f"  • Count: {data['count']} invoices\n"
                    response += f"  • Amount: SAR {data['amount']:,.2f}\n\n"
            
            response += f"**Total Outstanding:** SAR {report['total_outstanding']:,.2f}\n"
            response += f"**Total Invoices:** {report['total_invoices']}"
            
            return ChatResponse(
                response=response,
                data=report
            )
        
        # Handle payment reminders
        if action == "get_reminders":
            overdue = get_overdue_invoices()
            
            if not overdue:
                return ChatResponse(
                    response="✅ No overdue invoices! All payments are current.",
                    data=None
                )
            
            response = f"**{len(overdue)} Overdue Invoices Needing Follow-up**\n\n"
            for i, inv in enumerate(overdue[:10], 1):
                response += f"{i}. **{inv['invoice_number']}** - {inv['customer_name']}\n"
                response += f"   Amount: SAR {inv['balance']:,.2f} | {inv['days_overdue']} days overdue\n\n"
            
            total_overdue = sum(inv['balance'] for inv in overdue)
            response += f"**Total Overdue Amount:** SAR {total_overdue:,.2f}"
            
            return ChatResponse(
                response=response,
                data={'overdue_invoices': overdue}
            )
        
        # Get limits
        limits = get_limit_from_scope(scope, requested_count)
        fetch_limit = limits['fetch']
        display_limit = limits['display']
        
        # Execute API call based on action
        data = None
        pagination_info = None
        query_context = {
            'action': action,
            'intent': intent,
            'scope': scope,
            'per_page': fetch_limit,
            'customer_name': customer_name
        }
        
        if action == "get_invoices":
            data = get_invoices(customer_name=customer_name, limit=fetch_limit, page=page)
        elif action == "get_invoices_unpaid":
            data = get_invoices(status="unpaid", customer_name=customer_name, limit=fetch_limit, page=page)
        elif action == "get_quotes":
            data = get_quotes(customer_name=customer_name, limit=fetch_limit, page=page)
        elif action == "get_salesorders":
            data = get_salesorders(customer_name=customer_name, limit=fetch_limit, page=page)
        elif action == "get_customers":
            data = get_customers(limit=fetch_limit, page=page)
        elif action == "get_expenses":
            data = get_expenses(limit=fetch_limit, page=page)
        elif action == "get_bills":
            data = get_bills(limit=fetch_limit, page=page)
        elif action == "get_payments":
            data = get_payments(customer_name=customer_name, limit=fetch_limit, page=page)
        elif action == "get_recurring":
            data = get_recurring_invoices(customer_name=customer_name, limit=fetch_limit)
        elif action == "get_credits":
            data = get_credit_notes(customer_name=customer_name, limit=fetch_limit)
        elif action == "get_revenue":
            data = get_reports_revenue()
        else:
            return ChatResponse(
                response="I can help you with invoices, quotes, orders, bills, payments, expenses, customers, recurring invoices, credit notes, aging reports, and payment reminders. What would you like to know?",
                data=None
            )
        
        if data is None:
            return ChatResponse(
                response="❌ Sorry, I couldn't fetch data from Zoho Books. Please check your connection.",
                data=None
            )
        
        # Calculate pagination
        if action in ["get_invoices", "get_invoices_unpaid", "get_quotes", "get_salesorders", "get_customers", "get_expenses", "get_bills", "get_payments"]:
            pagination_info = calculate_pagination(data, page, fetch_limit)
        
        # Format response
        formatted_response = format_response_with_ai(data, intent, customer_name, display_limit)
        
        # Add pagination hint
        if pagination_info and (pagination_info.has_next or pagination_info.has_prev):
            formatted_response += f"\n\n**Page {pagination_info.current_page} of {pagination_info.total_pages}** (Total: {pagination_info.total_items} items)"
        
        return ChatResponse(
            response=formatted_response,
            data=data,
            pagination=pagination_info,
            query_context=query_context
        )
        
    except ZohoAPIError as e:
        return ChatResponse(
            response=f"❌ {e.message}",
            data=None
        )
    except Exception as e:
        return ChatResponse(
            response=f"❌ An error occurred: {str(e)}",
            data=None
        )

@app.get("/api/cache/clear")
async def clear_cache():
    """Clear the cache"""
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    zoho_cache.clear()
    return {"message": "Cache cleared successfully"}

@app.get("/test")
async def test_connection():
    """Test endpoint"""
    if not access_token:
        return {"status": "error", "message": "Not authenticated"}
    
    try:
        data = get_invoices(limit=1)
        return {"status": "success" if data else "error", "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)