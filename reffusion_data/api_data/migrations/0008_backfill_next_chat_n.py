from django.db import migrations

from api_data.chat_id_utils import next_chat_n_for


def backfill_next_chat_n(apps, schema_editor):
    Account = apps.get_model('api_data', 'Account')

    for account in Account.objects.all():
        chat_ids = list(account.chat.values_list('chat_id', flat=True))
        account.next_chat_n = next_chat_n_for(chat_ids)
        account.save(update_fields=['next_chat_n'])


def noop_reverse(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ('api_data', '0007_account_next_chat_n'),
    ]

    operations = [
        migrations.RunPython(backfill_next_chat_n, noop_reverse),
    ]
