from django.shortcuts import render
from django.http import JsonResponse
from django.db.models import Count, Avg, Q, Max, Min
from django.db.models.functions import Coalesce
from .models import Plant
from collections import Counter
import math

def plant_dashboard(request):
    """Main dashboard view"""
    return render(request, 'plants/dashboard.html')


def dashboard_stats_api(request):
    """Get overall dashboard statistics"""
    stats = {
        'total_plants': Plant.objects.count(),
        'total_families': Plant.objects.values('Family').distinct().count(),
        'total_genera': Plant.objects.values('Genus').distinct().count(),
        'total_orders': Plant.objects.values('Order').distinct().count(),
    }
    return JsonResponse(stats)


def dashboard_overview_api(request):
    """Get data for overview charts"""
    
    # Top 10 Families
    top_families = list(
        Plant.objects.values('Family')
        .annotate(count=Count('id'))
        .order_by('-count')[:10]
        .values('Family', 'count')
    )
    top_families_formatted = [{'label': f['Family'], 'count': f['count']} for f in top_families]
    
    # Life Form Distribution (Perennial, Woody, Agricultural)
    total = Plant.objects.count()
    life_forms = []
    
    # Count plants by life form percentages
    perennial_count = Plant.objects.filter(perc_per__gte=0.5).count()
    woody_count = Plant.objects.filter(perc_wood__gte=0.5).count()
    agricultural_count = Plant.objects.filter(perc_ag__gte=0.5).count()
    other_count = total - (perennial_count + woody_count + agricultural_count)
    
    if perennial_count > 0:
        life_forms.append({'label': 'Perennial', 'count': perennial_count})
    if woody_count > 0:
        life_forms.append({'label': 'Woody', 'count': woody_count})
    if agricultural_count > 0:
        life_forms.append({'label': 'Agricultural', 'count': agricultural_count})
    if other_count > 0:
        life_forms.append({'label': 'Other', 'count': other_count})
    
    # Hybridization Propensity Distribution
    hyb_data = Plant.objects.filter(HybProp__isnull=False).values_list('HybProp', flat=True)
    hybridization_dist = create_histogram(hyb_data, bins=10, range_min=0, range_max=1)
    
    # Genome Size Distribution
    genome_data = Plant.objects.filter(C_value__isnull=False).values_list('C_value', flat=True)
    genome_dist = create_histogram(genome_data, bins=15)
    
    return JsonResponse({
        'top_families': top_families_formatted,
        'life_forms': life_forms,
        'hybridization_dist': hybridization_dist,
        'genome_dist': genome_dist
    })


def dashboard_distribution_api(request):
    """Get data for distribution charts"""
    
    # Temperature Distribution
    temp_data = Plant.objects.filter(tavg__isnull=False).values_list('tavg', flat=True)
    temperature_dist = create_histogram(temp_data, bins=12)
    
    # Family Counts (Top 20 for heatmap)
    family_counts = list(
        Plant.objects.values('Family')
        .annotate(count=Count('id'))
        .order_by('-count')[:20]
    )
    
    # Order Distribution (Top 15)
    order_dist = list(
        Plant.objects.values('Order')
        .annotate(count=Count('id'))
        .order_by('-count')[:15]
    )
    order_dist_formatted = [{'label': o['Order'], 'count': o['count']} for o in order_dist]
    
    return JsonResponse({
        'temperature_dist': temperature_dist,
        'family_counts': family_counts,
        'order_dist': order_dist_formatted,
        'total_plants': Plant.objects.count()
    })


def dashboard_breeding_api(request):
    """Get data for breeding traits charts"""
    
    # Pollination Syndromes
    pollination = get_text_field_distribution('pollination_syndrome_text')
    
    # Mating Systems
    mating_systems = get_text_field_distribution('mating_system_text')
    
    # Floral Symmetry
    floral_symmetry = get_text_field_distribution('floral_symm_text')
    
    # Reproductive Syndromes
    reproductive_syndromes = get_text_field_distribution('repro_syndrome_text')
    
    # Family Compatibility (Average hybridization by top families)
    family_compatibility = list(
        Plant.objects.filter(HybProp__isnull=False)
        .values('Family')
        .annotate(
            count=Count('id'),
            avg_hyb=Avg('HybProp')
        )
        .filter(count__gte=10)  # Only families with 10+ plants
        .order_by('-avg_hyb')[:15]
    )
    
    family_compat_formatted = [
        {
            'label': f['Family'],
            'value': round(f['avg_hyb'], 3)
        }
        for f in family_compatibility
    ]
    
    return JsonResponse({
        'pollination': pollination,
        'mating_systems': mating_systems,
        'floral_symmetry': floral_symmetry,
        'reproductive_syndromes': reproductive_syndromes,
        'family_compatibility': family_compat_formatted
    })


def dashboard_conservation_api(request):
    """Get data for conservation charts"""
    
    # Red List Status
    red_list = get_text_field_distribution('redlist_text')
    
    # Agricultural Importance
    agricultural_data = [
        {'label': 'Agricultural', 'count': Plant.objects.filter(perc_ag__gte=0.5).count()},
        {'label': 'Non-Agricultural', 'count': Plant.objects.filter(Q(perc_ag__lt=0.5) | Q(perc_ag__isnull=True)).count()}
    ]
    
    # Conservation Insights
    insights = {
        'threatened': Plant.objects.filter(
            redlist_text__in=['Endangered', 'Critically Endangered', 'Vulnerable']
        ).count(),
        'agricultural': Plant.objects.filter(perc_ag__gte=0.5).count(),
        'woody': Plant.objects.filter(perc_wood__gte=0.5).count()
    }
    
    return JsonResponse({
        'red_list': red_list,
        'agricultural': agricultural_data,
        'insights': insights
    })


# Helper functions

def create_histogram(data, bins=10, range_min=None, range_max=None):
    """Create histogram data from a queryset of values"""
    if not data:
        return []
    
    data_list = list(data)
    if not data_list:
        return []
    
    if range_min is None:
        range_min = min(data_list)
    if range_max is None:
        range_max = max(data_list)
    
    # Create bins
    bin_width = (range_max - range_min) / bins
    histogram = []
    
    for i in range(bins):
        bin_start = range_min + (i * bin_width)
        bin_end = bin_start + bin_width
        
        # Count values in this bin
        count = sum(1 for val in data_list if bin_start <= val < bin_end or (i == bins - 1 and val == bin_end))
        
        # Format range label
        if bin_width < 1:
            range_label = f"{bin_start:.2f}-{bin_end:.2f}"
        else:
            range_label = f"{bin_start:.1f}-{bin_end:.1f}"
        
        histogram.append({
            'range': range_label,
            'count': count
        })
    
    return histogram


def get_text_field_distribution(field_name):
    """Get distribution of a text field, filtering out None/empty values"""
    distribution = (
        Plant.objects
        .filter(**{f'{field_name}__isnull': False})
        .exclude(**{f'{field_name}': ''})
        .values(field_name)
        .annotate(count=Count('id'))
        .order_by('-count')[:10]  # Top 10
    )
    
    return [
        {'label': item[field_name], 'count': item['count']}
        for item in distribution
        if item[field_name]  # Extra safety check
    ]