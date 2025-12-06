from django.db import models
from django.contrib.auth.models import AbstractUser

# Create your models here.
class CustomUser(AbstractUser):
    fullname = models.CharField(max_length=255)
    farm = models.CharField(max_length=255)
    region = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=20)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

    def __str__(self):
        return self.email
    
class Plant(models.Model):
    Genus = models.CharField(max_length=100)
    Family = models.CharField(max_length=100)
    Order = models.CharField(max_length=100)
    
    HybProp = models.FloatField(null=True, blank=True)
    Hyb_Ratio = models.FloatField(null=True, blank=True)
    perc_per = models.FloatField(null=True, blank=True)
    perc_wood = models.FloatField(null=True, blank=True)
    perc_ag = models.FloatField(null=True, blank=True)
    floral_symm = models.FloatField(null=True, blank=True)
    mating_system = models.FloatField(null=True, blank=True)
    repro_syndrome = models.FloatField(null=True, blank=True)
    pollination_syndrome = models.FloatField(null=True, blank=True)
    RedList = models.FloatField(null=True, blank=True)
    tavg = models.FloatField(null=True, blank=True)
    C_value = models.FloatField(null=True, blank=True)
    Cv_C_value = models.FloatField(null=True, blank=True)
    
    perc_per_text = models.CharField(max_length=100, null=True, blank=True)
    perc_wood_text = models.CharField(max_length=100, null=True, blank=True)
    perc_ag_text = models.CharField(max_length=100, null=True, blank=True)
    floral_symm_text = models.CharField(max_length=100, null=True, blank=True)
    mating_system_text = models.CharField(max_length=100, null=True, blank=True)
    repro_syndrome_text = models.CharField(max_length=100, null=True, blank=True)
    pollination_syndrome_text = models.CharField(max_length=100, null=True, blank=True)
    redlist_text = models.CharField(max_length=100, null=True, blank=True)
    tavg_text = models.CharField(max_length=100, null=True, blank=True)
    cvalue_text = models.CharField(max_length=100, null=True, blank=True)
    cv_cvalue_text = models.CharField(max_length=100, null=True, blank=True)
    image_url = models.TextField(null=True, blank=True)
    def __str__(self):
        return self.Genus