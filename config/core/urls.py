from django.urls import path 
from . import views , comparison_views , dashboard_views , chatbot

app_name = 'core'

urlpatterns = [
    path('',views.index,name="home"),
    path('predictor/',views.predictor,name="predictor"),
    path('predict/', views.predict_view, name='predict'),    path('login/',views.login_view,name="login"),
    path('signup/',views.signup,name="signup"),
    path('logout/',views.logout_view,name="logout"),

    path('catalog/',views.catalog,name="catalog"),
    # path('comparison/',views.comparison,name="comparison"),
    path('dashboard/',views.dashboard,name="dashboard"),
    path('profile/',views.profile,name="profile"),

    path('catalog/<int:pk>/', views.plant_detail, name="plant_detail"),
    path('catalog/load-plants/', views.load_plants, name='load_plants'),

    path('comparison/', comparison_views.plant_comparison, name='plant_comparison'),
    path('comparison/', comparison_views.plant_comparison, name='comparison'),
    
    # AJAX API endpoints for lazy loading
    path('api/plants/search/', comparison_views.plant_search_api, name='plant_search_api'),
    path('api/plants/families/', comparison_views.get_families_api, name='get_families_api'),
    path('api/plants/<int:plant_id>/', comparison_views.get_plant_details_api, name='get_plant_details_api'),

    path('plants/dashboard/', dashboard_views.plant_dashboard, name='plant_dashboard'),
    path('dashboard/', dashboard_views.plant_dashboard, name='dashboard'),
    
    # Dashboard API endpoints
    path('api/plants/dashboard/stats/', dashboard_views.dashboard_stats_api, name='dashboard_stats_api'),
    path('api/plants/dashboard/overview/', dashboard_views.dashboard_overview_api, name='dashboard_overview_api'),
    path('api/plants/dashboard/distribution/', dashboard_views.dashboard_distribution_api, name='dashboard_distribution_api'),
    path('api/plants/dashboard/breeding/', dashboard_views.dashboard_breeding_api, name='dashboard_breeding_api'),
    path('api/plants/dashboard/conservation/', dashboard_views.dashboard_conservation_api, name='dashboard_conservation_api'),
    
    path('chat-with-bot/', chatbot.chat_with_bot, name='chat_with_bot'),

]
