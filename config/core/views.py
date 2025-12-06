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
        plants = plants.filter(Family=family_filter)
    
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
    #from .gemini import HybridizationPredictor
    from .claude import HybridizationPredictor

except ImportError:
    HybridizationPredictor = None
    print("Warning: gemini.py not found. Prediction will not work.")

# Global storage to prevent reloading the heavy model on every request
# In a production app, you might use Django's cache or AppConfig.ready()
MODEL_CACHE = {
    'model': None,
    'data': None
}

# In your views.py:

def load_resources():
    if MODEL_CACHE['model'] is None and HybridizationPredictor:
        try:
            print("Loading Hybridization Model...")
            # Load the trained model artifact
            model = HybridizationPredictor.load_model('hybridization_predictor1.pkl')
            MODEL_CACHE['model'] = model

            # Correct path to enriched data (The lookup table)
            csv_path = settings.BASE_DIR / "data" / "data_clean.csv"
            
            # --- FIX STARTS HERE ---
            print("Loading FULL Database for lookup...")

            if not csv_path.exists():
                print("ERROR: data_clean.csv not found!")
                MODEL_CACHE['data'] = None
                return

            # Load ALL rows from the CSV, regardless of the train/test split.
            # The model is already trained on the split; the DF is only for UI lookup.
            df = pd.read_csv(csv_path)

            print("FULL CSV loaded successfully:", df.shape)

            MODEL_CACHE['data'] = df
            # --- FIX ENDS HERE ---

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




#! ##############################################################################


import json
import os
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from .models import Plant

# Load Algeria agro data
ALGERIA_DATA_PATH = os.path.join(settings.BASE_DIR, 'algeria_agro_data.json')

def load_algeria_data():
    """Load Algeria agricultural data from JSON file"""
    try:
        with open(ALGERIA_DATA_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def user_profile(request):
    """User profile page with plant recommendations"""
    algeria_data = load_algeria_data()
    
    # Get user's selected wilaya from session or query params
    selected_wilaya = request.GET.get('wilaya') or request.session.get('user_wilaya')
    
    context = {
        'algeria_data': algeria_data,
        'selected_wilaya': selected_wilaya
    }
    
    return render(request, 'core/profile.html', context)


def get_wilaya_recommendations_api(request):
    """API endpoint to get plant recommendations for a specific wilaya"""
    wilaya_name = request.GET.get('wilaya')
    
    if not wilaya_name:
        return JsonResponse({'error': 'Wilaya parameter required'}, status=400)
    
    # Load Algeria data
    algeria_data = load_algeria_data()
    
    # Find the wilaya data
    wilaya_info = None
    for region in algeria_data:
        if region['wilaya'].lower() == wilaya_name.lower():
            wilaya_info = region
            break
    
    if not wilaya_info:
        return JsonResponse({'error': 'Wilaya not found'}, status=404)
    
    # Extract environmental parameters
    air_temp = float(wilaya_info['temperature_celsius_air'])
    humidity = int(wilaya_info['humidity_percent_air'])
    soil_moisture = float(wilaya_info['soil_moisture_m3_m3'])
    soil_temp_surface = float(wilaya_info['soil_temperature_0cm'])
    soil_temp_10cm = float(wilaya_info['soil_temperature_10cm'])
    
    # Calculate average soil temperature
    avg_soil_temp = (soil_temp_surface + soil_temp_10cm) / 2
    
    # Determine climate classification
    climate_type = classify_climate(air_temp, humidity, soil_moisture)
    
    # Get recommended plants based on environmental conditions
    recommended_plants = find_compatible_plants(
        air_temp=air_temp,
        humidity=humidity,
        soil_moisture=soil_moisture,
        avg_soil_temp=avg_soil_temp
    )
    
    # Calculate compatibility scores
    plants_with_scores = []
    for plant in recommended_plants:
        score = calculate_compatibility_score(
            plant=plant,
            air_temp=air_temp,
            humidity=humidity,
            soil_moisture=soil_moisture,
            avg_soil_temp=avg_soil_temp
        )
        
        plants_with_scores.append({
            'id': plant.id,
            'genus': plant.Genus,
            'family': plant.Family,
            'order': plant.Order,
            'image_url': plant.image_url,
            'compatibility_score': round(score, 1),
            'temp_match': abs(plant.tavg - air_temp) if plant.tavg else None,
            'reasons': get_compatibility_reasons(plant, wilaya_info, score)
        })
    
    # Sort by compatibility score
    plants_with_scores.sort(key=lambda x: x['compatibility_score'], reverse=True)
    
    return JsonResponse({
        'wilaya': wilaya_info,
        'climate_type': climate_type,
        'recommended_plants': plants_with_scores[:50],  # Top 50 recommendations
        'total_compatible': len(plants_with_scores),
        'environmental_summary': {
            'air_temperature': air_temp,
            'humidity': humidity,
            'soil_moisture': soil_moisture,
            'avg_soil_temp': avg_soil_temp
        }
    })


def classify_climate(air_temp, humidity, soil_moisture):
    """Classify climate type based on environmental parameters"""
    if humidity < 30:
        if air_temp > 20:
            return {'type': 'Hot Desert', 'icon': '🏜️', 'description': 'Very hot and dry conditions'}
        else:
            return {'type': 'Cold Desert', 'icon': '🌵', 'description': 'Arid with moderate temperatures'}
    elif humidity < 50:
        if air_temp > 15:
            return {'type': 'Semi-Arid', 'icon': '🌾', 'description': 'Dry with warm temperatures'}
        else:
            return {'type': 'Cool Semi-Arid', 'icon': '🍂', 'description': 'Dry with cool temperatures'}
    elif humidity < 70:
        if air_temp > 18:
            return {'type': 'Mediterranean', 'icon': '🌿', 'description': 'Warm with moderate humidity'}
        else:
            return {'type': 'Temperate', 'icon': '🍃', 'description': 'Moderate temperatures and humidity'}
    else:
        if air_temp > 20:
            return {'type': 'Humid Subtropical', 'icon': '🌴', 'description': 'Warm and humid'}
        else:
            return {'type': 'Oceanic', 'icon': '🌊', 'description': 'Cool and humid'}

from django.db.models import Min, Max

# fetch min/max once
def get_stats():
    return Plant.objects.aggregate(
        tmin=Min('tavg'), tmax=Max('tavg'),
        wood_min=Min('perc_wood'), wood_max=Max('perc_wood'),
        per_min=Min('perc_per'), per_max=Max('perc_per'),
        ag_min=Min('perc_ag'), ag_max=Max('perc_ag'),
    )

def norm(value, vmin, vmax):
    if value is None or vmin is None or vmax is None or vmax == vmin:
        return 0
    return (value - vmin) / (vmax - vmin)


def find_compatible_plants(air_temp, humidity, soil_moisture, avg_soil_temp):
    stats = get_stats()

    # dynamic temp tolerance (5% of range or min ±3°C)
    temp_range = (stats["tmax"] - stats["tmin"]) or 10
    temp_tol = max(3, temp_range * 0.05)

    qs = Plant.objects.filter(tavg__isnull=False)

    # temp filter
    qs = qs.filter(
        tavg__gte=air_temp - temp_tol,
        tavg__lte=air_temp + temp_tol
    )

    # dynamic woody/perennial thresholds
    wood_thr = stats["wood_min"] + (stats["wood_max"] - stats["wood_min"]) * 0.30
    per_thr = stats["per_min"] + (stats["per_max"] - stats["per_min"]) * 0.50

    if soil_moisture < 0.1:
        qs = qs.filter(perc_wood__gte=wood_thr) | qs.filter(perc_per__gte=per_thr)

    elif soil_moisture > 0.3:
        qs = qs.exclude(perc_wood__gte=wood_thr * 1.5)

    return qs.distinct()[:100]



def calculate_compatibility_score(plant, air_temp, humidity, soil_moisture, avg_soil_temp):
    stats = get_stats()
    score = 100

    # ---- temp score (40 pts) ----
    temp_diff = abs((plant.tavg or air_temp) - air_temp)
    max_temp_diff = (stats["tmax"] - stats["tmin"]) or 10
    score -= (temp_diff / max_temp_diff) * 40

    # ---- moisture ----
    wood_n = norm(plant.perc_wood, stats["wood_min"], stats["wood_max"])
    per_n  = norm(plant.perc_per, stats["per_min"], stats["per_max"])
    ag_n   = norm(plant.perc_ag,  stats["ag_min"],  stats["ag_max"])

    if soil_moisture < 0.1:
        score += wood_n * 15
        score += per_n * 10
        score -= ag_n * 5

    elif soil_moisture > 0.3:
        score += (1 - wood_n) * 10
        score += ag_n * 10

    else:
        score += 10

    # ---- agricultural ----
    score += ag_n * 15

    # ---- hybridization ----
    if plant.HybProp:
        score += min(plant.HybProp, 1) * 10

    return int(max(0, min(100, score)))



def get_compatibility_reasons(plant, wilaya_info, score):
    reasons = []

    air_temp = float(wilaya_info["temperature_celsius_air"])
    soil_moisture = float(wilaya_info["soil_moisture_m3_m3"])

    # temp
    if plant.tavg:
        diff = abs(plant.tavg - air_temp)
        if diff <= 2:
            reasons.append("Excellent temperature match")
        elif diff <= 5:
            reasons.append("Good temperature tolerance")
        else:
            reasons.append("Requires temperature adaptation")

    # moisture
    if soil_moisture < 0.1 and plant.perc_wood and plant.perc_wood > 0.5:
        reasons.append("Drought-resistant woody plant")
    elif soil_moisture > 0.3:
        reasons.append("Thrives in moist conditions")

    # agri
    if plant.perc_ag and plant.perc_ag > 0.5:
        reasons.append("Proven agricultural crop")

    # breeding
    if plant.HybProp and plant.HybProp > 0.7:
        reasons.append("Easy to breed and adapt")

    # perennial
    if plant.perc_per and plant.perc_per > 0.7:
        reasons.append("Hardy perennial species")

    # final label
    if score >= 85:
        reasons.insert(0, "⭐ Highly recommended")
    elif score >= 70:
        reasons.insert(0, "✓ Well-suited")
    elif score >= 50:
        reasons.insert(0, "△ Possible with care")
    else:
        reasons.insert(0, "⚠ May be challenging")

    return reasons

def save_user_wilaya(request):
    """Save user's selected wilaya to session"""
    if request.method == 'POST':
        import json
        data = json.loads(request.body)
        wilaya = data.get('wilaya')
        
        if wilaya:
            request.session['user_wilaya'] = wilaya
            return JsonResponse({'success': True, 'wilaya': wilaya})
    
    return JsonResponse({'error': 'Invalid request'}, status=400)