from django.db import migrations


def create_demo_account(apps, schema_editor):
    Account = apps.get_model('api_data', 'Account')
    Chat = apps.get_model('api_data', 'Chat')

    account, _ = Account.objects.get_or_create(
        user_id='demo', defaults={'name': 'Demo User'}
    )
    Chat.objects.get_or_create(account=account, chat_id='main')


def remove_demo_account(apps, schema_editor):
    Account = apps.get_model('api_data', 'Account')
    Account.objects.filter(user_id='demo').delete()


class Migration(migrations.Migration):

    dependencies = [
        ('api_data', '0005_chatimages_chat_unique'),
    ]

    operations = [
        migrations.RunPython(create_demo_account, remove_demo_account),
    ]
