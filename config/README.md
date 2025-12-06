# AgroX - Agricultural Plant Intelligence System

## 📋 Overview

**AgroX** is a comprehensive agricultural intelligence platform built with Django that leverages data analytics, machine learning, and AI to provide intelligent plant recommendations and breeding compatibility analysis. The system helps farmers and agricultural professionals make data-driven decisions about plant selection, hybridization, and crop management.

### Key Features

- **🌱 Plant Catalog & Discovery**: Browse and search through an extensive database of plants with detailed characteristics
- **🔍 Advanced Plant Predictor**: AI-powered plant prediction based on environmental and botanical parameters
- **📊 Dashboard Analytics**: Comprehensive visualizations of plant data including breeding traits, distribution patterns, and conservation metrics
- **🤝 Plant Comparison Tool**: Compare multiple plants side-by-side to analyze compatibility and suitability
- **👤 User Profiles**: Personalized user experience with location-based (Wilaya) analysis and recommendations
- **💬 AI Chatbot**: Intelligent assistant for agricultural queries and plant-related information
- **🧬 Breeding Analysis**: Detailed compatibility scoring for plant hybridization with scientific metrics

---

## 🛠️ Technology Stack

### Backend
- **Framework**: Django 5.2.9
- **Database**: SQLite (development) / PostgreSQL (production recommended)
- **Python Version**: 3.8+
- **ORM**: Django ORM with custom models

### Frontend
- **Template Engine**: Django Templates (Jinja2-like)
- **CSS Framework**: Tailwind CSS
- **JavaScript Libraries**:
  - React (via CDN)
  - Recharts (for data visualization)
  - Lucide React (for icons)
  - React Router DOM

### ML & AI
- **Gemini API**: For AI-powered features and chatbot
- **Pandas**: Data processing and analysis
- **Python Pickle**: Model serialization for the Hybridization Predictor

### Additional Libraries
- `python-dotenv`: Environment variable management
- `django-extensions`: Django development utilities

---

## 📁 Project Structure

```
agrox/
├── config/                    # Django project configuration
│   ├── settings.py           # Django settings (database, apps, middleware)
│   ├── urls.py               # Project-level URL routing
│   ├── wsgi.py               # WSGI application
│   └── asgi.py               # ASGI application
│
├── core/                      # Main Django app
│   ├── models.py             # Database models (CustomUser, Plant)
│   ├── views.py              # Main views (auth, catalog, predictor)
│   ├── urls.py               # App-level URL routing
│   ├── admin.py              # Django admin configuration
│   ├── comparison_views.py   # Plant comparison logic
│   ├── dashboard_views.py    # Dashboard and analytics
│   ├── chatbot.py            # AI chatbot functionality
│   ├── gemini1.py            # Hybridization predictor ML model
│   ├── migrations/           # Database migration files
│   └── templates/            # HTML templates
│       ├── base.html         # Base template (navbar, layout)
│       └── core/
│           ├── index.html
│           ├── catalog.html
│           ├── plant-details.html
│           ├── comparison.html
│           ├── dashboard.html
│           ├── predictor.html
│           ├── profile.html
│           ├── login.html
│           ├── signup.html
│           └── partials/
│               ├── chatbot-widget.html
│               └── plant_cards.html
│
├── data/                      # Data files
│   └── data_clean.csv        # Cleaned plant data for ML model
│
├── manage.py                 # Django management script
├── db.sqlite3                # SQLite database (dev)
├── family_data.csv           # Plant family reference data
├── genus_data.csv            # Plant genus reference data
└── requirements.txt          # Python dependencies (to be created)
```

---

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.8 or higher
- **pip**: Python package manager
- **Git**: For version control
- **Virtual Environment**: Recommended (venv, conda, or similar)

### Installation Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/yacineLaribi/agrox.git
cd agrox
```

#### 2. Create and Activate Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

#### 3. Install Dependencies

First, create a `requirements.txt` file with the necessary packages:

```bash
pip install django==5.2.9
pip install pandas
pip install google-generativeai  # For Gemini API
pip install python-dotenv
pip install django-extensions
```

Or if a `requirements.txt` already exists:
```bash
pip install -r requirements.txt
```

#### 4. Environment Configuration

Create a `.env` file in the project root with the following variables:

```bash
# .env file
GOOGLE_API_KEY=your_google_gemini_api_key_here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
```

#### 5. Apply Database Migrations

```bash
python manage.py migrate
```

#### 6. Create a Superuser (for Django Admin)

```bash
python manage.py createsuperuser
```

Follow the prompts to create your admin account.

#### 7. Load Initial Data (Optional)

If you have data migration scripts:

```bash
python manage.py populate_plants
```

This loads plant data from CSV files into the database.

---

## ▶️ Running the Project

### Development Server

Start the Django development server:

```bash
python manage.py runserver
```

The application will be available at: **http://localhost:8000**

### Access the Admin Panel

Navigate to: **http://localhost:8000/admin/**

Log in with your superuser credentials to manage plants, users, and other data.

### Common Django Commands

```bash
# Make migrations after model changes
python manage.py makemigrations

# Apply pending migrations
python manage.py migrate

# Create a new superuser
python manage.py createsuperuser

# Collect static files (for production)
python manage.py collectstatic

# Run the development server on a specific port
python manage.py runserver 0.0.0.0:8080

# Access Django shell for interactive development
python manage.py shell
```

---

## 🗄️ Database Models

### CustomUser Model
Extends Django's AbstractUser with additional agricultural fields:
- `fullname`: User's full name
- `farm`: User's farm name
- `region`: Wilaya (Algerian region)
- `email`: User's email
- `phone`: Contact number

### Plant Model
Comprehensive plant data model with attributes including:
- **Basic Info**: Genus, Family, Order
- **Breeding Traits**: 
  - `HybProp`: Hybridization propensity
  - `Hyb_Ratio`: Hybridization ratio
  - `mating_system`: Sexual/asexual classification
  - `pollination_syndrome`: Pollination type
  - `repro_syndrome`: Reproduction syndrome
  - `floral_symm`: Floral symmetry (bilateral/radial)
- **Agricultural**: 
  - `perc_per`: Perennial percentage
  - `perc_wood`: Wood percentage
  - `perc_ag`: Agricultural use percentage
- **Climate Data**: 
  - `tavg`: Average temperature
  - `precip`: Average precipitation

---

## 🔑 Core Features Explained

### 1. **Plant Predictor** (`/predictor/`)
Uses the `HybridizationPredictor` ML model to predict plant compatibility:
- Analyzes breeding traits
- Provides compatibility scores
- Suggests optimal plant combinations

### 2. **Dashboard** (`/dashboard/`)
Interactive analytics with multiple chart types:
- **Overview Charts**: Global plant statistics
- **Distribution Charts**: Geographic and taxonomic distributions
- **Breeding Traits**: Pollination syndromes, mating systems, floral symmetry
- **Conservation Data**: Threatened species analysis
- **Family Compatibility**: Cross-family hybridization analysis

### 3. **Plant Comparison** (`/comparison/`)
Compare up to multiple plants with:
- Physical characteristics
- Breeding compatibility analysis
- Environmental requirements
- Favorable factors and concerns

### 4. **User Profile** (`/profile/`)
Location-based analysis:
- Set preferred Wilaya (Algerian region)
- Get personalized plant suitability recommendations
- Analyze regional agricultural opportunities

### 5. **Catalog** (`/catalog/`)
Browse complete plant database with:
- Infinite scroll pagination
- Search and filter capabilities
- Detailed plant information cards
- Links to individual plant details

### 6. **AI Chatbot**
Integrated chatbot using Gemini API:
- Answer agricultural questions
- Provide plant care recommendations
- Support for natural language queries

---

## 🔧 Configuration

### Django Settings (`config/settings.py`)
Key configurations:

```python
# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Installed Apps
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'core',
]

# Custom User Model
AUTH_USER_MODEL = 'core.CustomUser'
```

### Static Files & Media
- Static files location: `static/`
- Templates location: `core/templates/`

---

## 📊 API Endpoints

### Authentication
- `GET /` - Home page
- `POST /login/` - User login
- `POST /signup/` - User registration
- `GET /logout/` - User logout

### Plant Data
- `GET /catalog/` - Plant catalog with pagination
- `GET /catalog/<id>/` - Individual plant details
- `POST /catalog/load-plants/` - AJAX for infinite scroll

### Analytics
- `GET /dashboard/` - Dashboard overview
- `POST /dashboard/stats/` - Statistics API
- `POST /dashboard/overview/` - Overview charts data
- `POST /dashboard/distribution/` - Distribution data
- `POST /dashboard/breeding/` - Breeding traits data
- `POST /dashboard/conservation/` - Conservation data

### AI Features
- `GET /predictor/` - Plant predictor interface
- `POST /predict/` - Get predictions from ML model
- `POST /chatbot/` - Chatbot API endpoint

### Comparison
- `GET /comparison/` - Comparison interface
- `POST /comparison/analyze/` - Breeding compatibility analysis

### User
- `GET /profile/` - User profile with regional analysis

---

## 🧪 Testing

Run tests with:

```bash
python manage.py test
```

Run specific test file:

```bash
python manage.py test core.tests
```

Run with coverage (if coverage.py is installed):

```bash
pip install coverage
coverage run --source='.' manage.py test
coverage report
```

---

## 📦 Deployment

### Production Checklist

1. **Security Settings** (`config/settings.py`):
   ```python
   DEBUG = False
   ALLOWED_HOSTS = ['yourdomain.com', 'www.yourdomain.com']
   SECRET_KEY = os.environ.get('SECRET_KEY')  # Use secure key
   ```

2. **Database**: Switch from SQLite to PostgreSQL:
   ```python
   DATABASES = {
       'default': {
           'ENGINE': 'django.db.backends.postgresql',
           'NAME': os.environ.get('DB_NAME'),
           'USER': os.environ.get('DB_USER'),
           'PASSWORD': os.environ.get('DB_PASSWORD'),
           'HOST': os.environ.get('DB_HOST'),
           'PORT': os.environ.get('DB_PORT', '5432'),
       }
   }
   ```

3. **Static Files**:
   ```bash
   python manage.py collectstatic --noinput
   ```

4. **HTTPS**: Enable SSL/TLS in production

5. **Web Server**: Use Gunicorn or uWSGI
   ```bash
   pip install gunicorn
   gunicorn config.wsgi:application --bind 0.0.0.0:8000
   ```

6. **Reverse Proxy**: Configure Nginx or Apache

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'django'`
- **Solution**: Activate virtual environment and run `pip install -r requirements.txt`

**Issue**: Database locked error
- **Solution**: Delete `db.sqlite3` and run migrations again

**Issue**: Static files not loading
- **Solution**: Run `python manage.py collectstatic`

**Issue**: Gemini API key not found
- **Solution**: Ensure `.env` file is in project root with `GOOGLE_API_KEY` set

**Issue**: Plant predictor model not loading
- **Solution**: Ensure `hybridization_predictor.pkl` exists in project root and `data_clean.csv` exists in `data/` folder

---

## 📚 Documentation & Resources

- **Django Documentation**: https://docs.djangoproject.com/en/5.2/
- **Tailwind CSS**: https://tailwindcss.com/docs
- **Recharts**: https://recharts.org/
- **Google Gemini API**: https://ai.google.dev/

---

## 👥 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📧 Contact & Support

- **Project Maintainer**: Yacine Laribi
- **GitHub**: https://github.com/yacineLaribi/agrox
- **Issues**: Please report bugs and feature requests on GitHub Issues

---

## 🙏 Acknowledgments

- Django community for the excellent web framework
- Google Generative AI for the Gemini API
- Contributors and users who provide feedback and improvements
- Agricultural research data sources

---

## 📋 Quick Reference

### Environment Variables
```bash
GOOGLE_API_KEY=your_api_key
DEBUG=True/False
ALLOWED_HOSTS=localhost,127.0.0.1
SECRET_KEY=your_secret_key
```

### Useful Database Commands
```bash
# View all plants
python manage.py shell
>>> from core.models import Plant
>>> Plant.objects.all().count()

# Filter plants by genus
>>> Plant.objects.filter(Genus='Rosa')

# Update plant data
>>> p = Plant.objects.get(id=1)
>>> p.HybProp = 0.85
>>> p.save()
```

---

**Last Updated**: December 6, 2025  
**Version**: 1.0.0  
**Status**: Production Ready

---

*For additional help or questions, please check the GitHub repository or contact the development team.*
