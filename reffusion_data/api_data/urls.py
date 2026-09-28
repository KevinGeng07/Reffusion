from django.urls import path
from .views import *

urlpatterns = [
    path('login/', LoginView.as_view(), name='login'),
    path('signup/', signup, name='signup'),
    path('logout/', logout, name='logout'),
    path('models/', list_models, name='list_models'),
    path('models/<str:model_key>/warm/', warm_model, name='warm_model'),
    path('accounts/', get_accounts, name='get_accounts'),
    path('accounts/<str:user_id>/', get_account, name='get_account'),
    path('accounts/<str:user_id>/new_chat/', create_chat, name='create_chat'),
    path('accounts/<str:user_id>/chat/<str:chat_id>/', chat_detail, name='chat_detail'),
]