from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api_data', '0006_seed_demo_account'),
    ]

    operations = [
        migrations.AddField(
            model_name='account',
            name='next_chat_n',
            field=models.PositiveIntegerField(default=1),
        ),
    ]
