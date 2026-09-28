from rest_framework import serializers
from .models import Account, Chat, ChatImages


class ChatImagesSerial(serializers.ModelSerializer):
    class Meta:
        model = ChatImages
        fields = ['log_id', 'text', 'image', 'comparison_image']
        read_only_fields = ['log_id', 'chat']

class ChatSerial(serializers.ModelSerializer):
    chat_log = ChatImagesSerial(many=True, read_only=True)

    class Meta:
        model = Chat
        fields = [
            'chat_id', 'name', 'use_conditioning', 'conditioning_strength',
            'num_inference_steps', 'model_key', 'chat_log',
        ]
        read_only_fields = ['account', 'chat_id']

    def validate(self, data):
        # strength/steps combine as int(steps * strength) effective denoising
        # steps in img2img; at 0, diffusers crashes on an empty latent tensor
        # instead of a graceful no-op, so this combination has to be rejected
        # up front rather than left to fail at generation time.
        strength = data.get(
            'conditioning_strength',
            getattr(self.instance, 'conditioning_strength', 0.8),
        )
        steps = data.get(
            'num_inference_steps', getattr(self.instance, 'num_inference_steps', 4),
        )
        if int(steps * strength) < 1:
            raise serializers.ValidationError(
                f'strength ({strength}) x steps ({steps}) must be at least 1 '
                '- this combination would run 0 denoising steps.'
            )
        return data

class AccountSerial(serializers.ModelSerializer):
    chat = ChatSerial(many=True, read_only=True)

    class Meta:
        model = Account
        fields = ['name', 'user_id', 'chat']