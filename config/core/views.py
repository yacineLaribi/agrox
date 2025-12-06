from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login
from django.contrib import messages
from core.models import CustomUser as User
from django.contrib.auth import logout
from django.core.paginator import Paginator
from core.models import Plant
# Create your views here.
#! Authentication views 
def signup(request):
    if request.method == "POST":
        farm = request.POST.get("farm")
        fullname = request.POST.get("fullname")
        email = request.POST.get("email")
        region = request.POST.get("region")
        password = request.POST.get("password")
        password2 = request.POST.get("password2")
        
        if password != password2:
            messages.error(request, "Passwords do not match.")
            return render(request, "core/signup.html")
        
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered.")
            return render(request, "core/signup.html")
        
        user = User.objects.create_user(username=email,farm=farm,fullname=fullname,region=region, email=email, password=password)
        login(request, user)
        messages.success(request, "Welcome to the intelligent world of Agriculture with Jnan.")

        return redirect("core:home")
    
    return render(request, "core/signup.html")


def login_view(request):
    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")

        if not email or not password:
            messages.error(request, "Please provide email and password.")
            return render(request, "core/login.html", {})


        # Try authenticating assuming username == email (signup uses username=email)
        user = authenticate(request, username=email, password=password)

        # Fallback: if your user model uses a different username field, try resolving by email
        if user is None:
            try:
                u = User.objects.get(email=email)
                user = authenticate(request, username=u.username, password=password)
            except User.DoesNotExist:
                user = None

        if user is not None:
            login(request, user)
            display_name = getattr(user, 'fullname', None) or (user.get_full_name() if callable(getattr(user, 'get_full_name', None)) else None) or user.username
            messages.success(request, f"Welcome back, {display_name}!")
            return redirect("core:home")

        messages.error(request, "Login failed. Check email and password.")

    return render(request, "core/login.html", {})

def logout_view(request):
    logout(request)
    return redirect("core:home")


def index(request):
    context = {}
    return render(request,"core/index.html",context)



def dashboard(request):
    context = {}
    return render(request,"core/dashboard.html",context)

def profile(request):
    context = {}
    return render(request,"core/profile.html",context)


from django.core.paginator import Paginator
from django.shortcuts import render
from django.http import JsonResponse
from django.db.models import Q
from django.template.loader import render_to_string

def catalog(request):
    """Main catalog page"""
    families = Plant.objects.values_list('Family', flat=True).distinct().order_by('Family')
    context = {'families': families}
    return render(request, "core/catalog.html", context)

def load_plants(request):
    """AJAX endpoint for infinite scroll"""
    page = int(request.GET.get('page', 1))
    search_query = request.GET.get('search', '')
    family_filter = request.GET.get('Family', '')
    
    # Order by id to prevent pagination warnings
    plants = Plant.objects.exclude(image_url__isnull=True).exclude(image_url__exact='No image found').order_by('id')
    
    # Apply filters
    if search_query:
        plants = plants.filter(
            Q(Genus__icontains=search_query) | 
            Q(Family__icontains=search_query)
        )
    
    if family_filter:
        plants = plants.filter(family=family_filter)
    
    # Paginate
    paginator = Paginator(plants, 10)  # 20 plants per load
    page_obj = paginator.get_page(page)
    
    # Render plant cards HTML
    plants_html = render_to_string('core/partials/plant_cards.html', {
        'plants': page_obj
    })
    
    return JsonResponse({
        'html': plants_html,
        'has_next': page_obj.has_next(),
        'next_page': page_obj.next_page_number() if page_obj.has_next() else None,
        'total_count': paginator.count
    })

from django.shortcuts import render, get_object_or_404
from django.db.models import Q

def plant_detail(request, pk):

    """Plant detail page"""
    plant = get_object_or_404(Plant, pk=pk)
    
    # Get related plants from the same family
    related_plants = Plant.objects.filter(
        Family=plant.Family
    ).exclude(pk=plant.pk)[:4]
    
    # Get potential hybridization candidates (same family, high hybrid propensity)
    hybrid_candidates = Plant.objects.filter(
        Family=plant.Family,
        HybProp__gte=0.5
    ).exclude(pk=plant.pk)[:3] if plant.HybProp else []
    
    context = {
        'plant': plant,
        'related_plants': related_plants,
        'hybrid_candidates': hybrid_candidates,
    }
    return render(request, "core/plant-details.html", context)

# def predictor(request):
#     context = {}
#     return render(request,"core/predictor.html",context)
import json
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from django.apps import apps
from django.conf import settings

# Import your predictor class
# Ensure 'gemini.py' is in the same directory or properly on your PYTHONPATH
try:
    from .gemini import HybridizationPredictor
except ImportError:
    HybridizationPredictor = None
    print("Warning: gemini.py not found. Prediction will not work.")

# Global storage to prevent reloading the heavy model on every request
# In a production app, you might use Django's cache or AppConfig.ready()
MODEL_CACHE = {
    'model': None,
    'data': None
}

def load_resources():
    if MODEL_CACHE['model'] is None and HybridizationPredictor:
        try:
            print("Loading Hybridization Model and Database...")
            model = HybridizationPredictor.load_model('hybridization_predictor.pkl')
            MODEL_CACHE['model'] = model

            # Correct path to enriched data
            csv_path = settings.BASE_DIR / "data" / "data_clean.csv"

            print("Looking for CSV at:", csv_path)

            if not csv_path.exists():
                print("ERROR: data_clean.csv not found!")
                MODEL_CACHE['data'] = None
                return

            df = pd.read_csv(csv_path)

            print("CSV loaded successfully:", df.shape)

            MODEL_CACHE['data'] = df

        except Exception as e:
            print("Load error:", e)
            MODEL_CACHE['data'] = None

from django.apps import apps
from django.shortcuts import render
from django.contrib.auth import authenticate

def predictor(request):
    """Renders the frontend with the dropdown options, fetching genera from the core.Plant model."""
    load_resources()
    
    genera_list = []
    try:
        # Get the Plant model dynamically from the 'core' app
        Plant = apps.get_model('core', 'Plant')
        
        # Query database for id + Genus
        genera_list = list(
            Plant.objects
            .values('id', 'Genus')
            .exclude(Genus__isnull=True)
            .exclude(Genus='')
            .order_by('Genus')
        )
    except LookupError:
        print("Error: Could not find 'core.Plant' model. Ensure the 'core' app is installed and the model exists.")
        # Fallback to DataFrame if model not available
        if MODEL_CACHE['data'] is not None and 'Genus' in MODEL_CACHE['data'].columns:
            genera_list = [
                {'id': idx, 'Genus': name} 
                for idx, name in enumerate(sorted(MODEL_CACHE['data']['Genus'].dropna().unique().tolist()))
            ]
    except Exception as e:
        print(f"Database query error in predictor view: {e}")
    
    # Render template
    return render(request, 'core/predictor.html', {'genera': genera_list})

def predict_view(request):
    """API Endpoint: Receives 2 genera, returns ML prediction + plant details."""
    if request.method == 'POST':
        load_resources()
        
        if not MODEL_CACHE['model']:
            return JsonResponse({'error': 'Model could not be loaded on server.'}, status=503)

        try:
            data = json.loads(request.body)
            g1 = data.get('genus1')
            g2 = data.get('genus2')
            
            if not g1 or not g2:
                return JsonResponse({'error': 'Please select both genera.'}, status=400)

            # 1. Run the ML Prediction
            # Result contains: {'prediction': X, 'probability': Y, 'confidence': Z}
            prediction_result = MODEL_CACHE['model'].predict(g1, g2)
            
            # 2. Fetch details for the UI cards (Family, Order, C-value, etc.)
            # We look up the original data rows to populate the sidebar info
            plant1_info = {}
            plant2_info = {}
            
            df = MODEL_CACHE['data']
            if df is not None:
                # Helper to get dict from dataframe row
                def get_row_data(genus):
                    row = df[df['Genus'] == genus]
                    if not row.empty:
                        data = row.iloc[0].where(pd.notnull(row.iloc[0]), None).to_dict()
                        # ensure image is present in the dict
                        data['image_url'] = data.get('image_url')
                        return data
                    return {}


                plant1_info = get_row_data(g1)
                plant2_info = get_row_data(g2)

            return JsonResponse({
                'success': True,
                'result': prediction_result,
                'plant1': plant1_info,
                'plant2': plant2_info
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)


