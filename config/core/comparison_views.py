from django.shortcuts import render
from django.http import JsonResponse
from django.core.paginator import Paginator
from django.db.models import Q
from .models import Plant

def plant_comparison(request):
    """View for plant comparison page"""
    # Get selected plant IDs from GET parameters
    selected_ids = request.GET.getlist('plants')
    selected_plants = []
    
    if selected_ids:
        selected_plants = Plant.objects.filter(id__in=selected_ids)[:3]  # Max 3 plants
    
    context = {
        'selected_plants': selected_plants,
        'breeding_analysis': None
    }
    
    # If we have multiple plants selected, analyze breeding compatibility
    if len(selected_plants) >= 2:
        context['breeding_analysis'] = analyze_breeding_compatibility(list(selected_plants))
    
    return render(request, 'core/comparison.html', context)


def plant_search_api(request):
    """
    AJAX endpoint for searching and filtering plants
    Returns paginated, lightweight plant data
    """
    search_query = request.GET.get('search', '').strip()
    family_filter = request.GET.get('family', '').strip()
    page = int(request.GET.get('page', 1))
    per_page = 50  # Load 50 plants at a time
    
    # Start with all plants
    plants = Plant.objects.all()
    
    # Apply search filter
    if search_query:
        plants = plants.filter(
            Q(Genus__icontains=search_query) |
            Q(Family__icontains=search_query) |
            Q(Order__icontains=search_query)
        )
    
    # Apply family filter
    if family_filter:
        plants = plants.filter(Family=family_filter)
    
    # Order by genus
    plants = plants.order_by('Genus')
    
    # Paginate
    paginator = Paginator(plants, per_page)
    page_obj = paginator.get_page(page)
    
    # Return lightweight data
    plants_data = [{
        'id': plant.id,
        'genus': plant.Genus,
        'family': plant.Family,
        'order': plant.Order,
        'image_url': plant.image_url if plant.image_url else None
    } for plant in page_obj]
    
    return JsonResponse({
        'plants': plants_data,
        'has_next': page_obj.has_next(),
        'has_previous': page_obj.has_previous(),
        'current_page': page,
        'total_pages': paginator.num_pages,
        'total_count': paginator.count
    })


def get_families_api(request):
    """
    Get list of unique families for filter dropdown
    """
    families = Plant.objects.values_list('Family', flat=True).distinct().order_by('Family')
    return JsonResponse({
        'families': list(families)
    })


def get_plant_details_api(request, plant_id):
    """
    Get detailed info for a specific plant (when selected)
    """
    try:
        plant = Plant.objects.get(id=plant_id)
        return JsonResponse({
            'id': plant.id,
            'genus': plant.Genus,
            'family': plant.Family,
            'order': plant.Order,
            'image_url': plant.image_url,
            'hybridization_propensity': plant.HybProp,
            'c_value': plant.C_value,
        })
    except Plant.DoesNotExist:
        return JsonResponse({'error': 'Plant not found'}, status=404)


def analyze_breeding_compatibility(plants):
    """
    Analyze if plants can breed together based on their characteristics
    Returns a dictionary with compatibility analysis
    """
    analysis = {
        'can_breed': True,
        'compatibility_score': 100,
        'blocking_factors': [],
        'concerns': [],
        'favorable_factors': [],
        'detailed_analysis': {}
    }
    
    # 1. Check Taxonomic Compatibility
    genera = [p.Genus for p in plants]
    families = [p.Family for p in plants]
    orders = [p.Order for p in plants]
    
    if len(set(genera)) > 1:
        if len(set(families)) > 1:
            analysis['blocking_factors'].append({
                'factor': 'Different Families',
                'severity': 'Critical',
                'detail': f"Plants from different families ({', '.join(set(families))}) cannot interbreed."
            })
            analysis['can_breed'] = False
            analysis['compatibility_score'] = 0
        else:
            analysis['concerns'].append({
                'factor': 'Different Genera',
                'severity': 'High',
                'detail': f"Cross-genus breeding ({', '.join(set(genera))}) is extremely rare and difficult."
            })
            analysis['compatibility_score'] -= 40
    else:
        analysis['favorable_factors'].append({
            'factor': 'Same Genus',
            'detail': f"All plants belong to genus {genera[0]}, enabling potential hybridization."
        })
    
    # 2. Check Hybridization Propensity
    hyb_props = [p.HybProp for p in plants if p.HybProp is not None]
    if hyb_props:
        avg_hyb = sum(hyb_props) / len(hyb_props)
        if avg_hyb < 0.3:
            analysis['concerns'].append({
                'factor': 'Low Hybridization Propensity',
                'severity': 'Medium',
                'detail': f"Average hybridization propensity is {avg_hyb:.2f}, indicating low natural breeding tendency."
            })
            analysis['compatibility_score'] -= 20
        elif avg_hyb > 0.7:
            analysis['favorable_factors'].append({
                'factor': 'High Hybridization Propensity',
                'detail': f"Average hybridization propensity of {avg_hyb:.2f} suggests good breeding potential."
            })
    
    # 3. Check C-values (Genome Size)
    c_values = [p.C_value for p in plants if p.C_value is not None]
    if len(c_values) >= 2:
        max_c = max(c_values)
        min_c = min(c_values)
        c_diff_ratio = max_c / min_c if min_c > 0 else float('inf')
        
        if c_diff_ratio > 2.0:
            analysis['blocking_factors'].append({
                'factor': 'Incompatible Genome Sizes',
                'severity': 'Critical',
                'detail': f"Genome size difference of {c_diff_ratio:.1f}x ({min_c:.2f} to {max_c:.2f}) prevents successful breeding."
            })
            analysis['can_breed'] = False
            analysis['compatibility_score'] = max(0, analysis['compatibility_score'] - 50)
        elif c_diff_ratio > 1.5:
            analysis['concerns'].append({
                'factor': 'Moderate Genome Size Difference',
                'severity': 'Medium',
                'detail': f"Genome sizes vary by {c_diff_ratio:.1f}x, which may reduce hybrid viability."
            })
            analysis['compatibility_score'] -= 15
        else:
            analysis['favorable_factors'].append({
                'factor': 'Compatible Genome Sizes',
                'detail': f"Similar genome sizes (ratio {c_diff_ratio:.2f}x) favor successful hybridization."
            })
    
    # 4. Check Mating Systems
    mating_systems = [p.mating_system_text for p in plants if p.mating_system_text]
    if len(set(mating_systems)) > 1:
        analysis['concerns'].append({
            'factor': 'Different Mating Systems',
            'severity': 'Low',
            'detail': f"Varying mating systems ({', '.join(set(mating_systems))}) may complicate breeding efforts."
        })
        analysis['compatibility_score'] -= 10
    
    # 5. Check Pollination Syndromes
    poll_syndromes = [p.pollination_syndrome_text for p in plants if p.pollination_syndrome_text]
    if len(set(poll_syndromes)) > 1:
        analysis['concerns'].append({
            'factor': 'Different Pollination Syndromes',
            'severity': 'Medium',
            'detail': f"Different pollination mechanisms ({', '.join(set(poll_syndromes))}) may require manual intervention."
        })
        analysis['compatibility_score'] -= 15
    else:
        if poll_syndromes:
            analysis['favorable_factors'].append({
                'factor': 'Compatible Pollination',
                'detail': f"Shared pollination syndrome ({poll_syndromes[0]}) facilitates cross-pollination."
            })
    
    # 6. Check Floral Symmetry
    floral_symms = [p.floral_symm_text for p in plants if p.floral_symm_text]
    if len(set(floral_symms)) > 1:
        analysis['concerns'].append({
            'factor': 'Different Floral Symmetry',
            'severity': 'Low',
            'detail': f"Different floral structures ({', '.join(set(floral_symms))}) may affect pollination success."
        })
        analysis['compatibility_score'] -= 5
    
    # 7. Check Temperature Requirements (tavg)
    tavg_values = [p.tavg for p in plants if p.tavg is not None]
    if len(tavg_values) >= 2:
        max_temp = max(tavg_values)
        min_temp = min(tavg_values)
        temp_diff = max_temp - min_temp
        
        if temp_diff > 10:
            analysis['concerns'].append({
                'factor': 'Different Climate Requirements',
                'severity': 'Medium',
                'detail': f"Temperature preference difference of {temp_diff:.1f}°C may indicate ecological isolation."
            })
            analysis['compatibility_score'] -= 10
    
    # Ensure score doesn't go below 0
    analysis['compatibility_score'] = max(0, analysis['compatibility_score'])
    
    # Generate overall recommendation
    if analysis['compatibility_score'] >= 70:
        analysis['recommendation'] = 'Good Breeding Potential'
        analysis['recommendation_class'] = 'success'
    elif analysis['compatibility_score'] >= 40:
        analysis['recommendation'] = 'Possible with Intervention'
        analysis['recommendation_class'] = 'warning'
    else:
        analysis['recommendation'] = 'Not Recommended'
        analysis['recommendation_class'] = 'danger'
    
    return analysis