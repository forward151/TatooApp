# Generated by Django 2.2.19 on 2023-05-31 15:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0004_remove_data_image'),
    ]

    operations = [
        migrations.AddField(
            model_name='data',
            name='image_name',
            field=models.TextField(default=0, max_length=50),
            preserve_default=False,
        ),
    ]
