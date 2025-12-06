import csv
from django.core.management.base import BaseCommand
from core.models import Plant

class Command(BaseCommand):
    help = "Populate the Plant table from CSV file"

    def add_arguments(self, parser):
        parser.add_argument(
            '--file',
            type=str,
            default='data/plants.csv',
            help='Path to the CSV file to import'
        )

    def handle(self, *args, **options):
        file_path = options['file']
        self.stdout.write(f"Loading plants from {file_path}...")

        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            count = 0
            for row in reader:
                # convert numeric fields safely
                def to_float(val):
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return None

                plant, created = Plant.objects.update_or_create(
                    Genus=row['Genus'],
                    defaults={
                        'Family': row.get('Family', ''),
                        'Order': row.get('Order', ''),
                        'HybProp': to_float(row.get('HybProp')),
                        'Hyb_Ratio': to_float(row.get('Hyb_Ratio')),
                        'perc_per': to_float(row.get('perc_per')),
                        'perc_wood': to_float(row.get('perc_wood')),
                        'perc_ag': to_float(row.get('perc_ag')),
                        'floral_symm': to_float(row.get('floral_symm')),
                        'mating_system': to_float(row.get('mating_system')),
                        'repro_syndrome': to_float(row.get('repro_syndrome')),
                        'pollination_syndrome': to_float(row.get('pollination_syndrome')),
                        'RedList': to_float(row.get('RedList')),
                        'tavg': to_float(row.get('tavg')),
                        'C_value': to_float(row.get('C_value')),
                        'Cv_C_value': to_float(row.get('Cv_C_value')),
                        'perc_per_text': row.get('perc_per_text', ''),
                        'perc_wood_text': row.get('perc_wood_text', ''),
                        'perc_ag_text': row.get('perc_ag_text', ''),
                        'floral_symm_text': row.get('floral_symm_text', ''),
                        'mating_system_text': row.get('mating_system_text', ''),
                        'repro_syndrome_text': row.get('repro_syndrome_text', ''),
                        'pollination_syndrome_text': row.get('pollination_syndrome_text', ''),
                        'redlist_text': row.get('redlist_text', ''),
                        'tavg_text': row.get('tavg_text', ''),
                        'cvalue_text': row.get('cvalue_text', ''),
                        'cv_cvalue_text': row.get('cv_cvalue_text', ''),
                        'image_url': row.get('image_url', ''),
                    }
                )
                count += 1

        self.stdout.write(self.style.SUCCESS(f"Imported/updated {count} plants!"))
