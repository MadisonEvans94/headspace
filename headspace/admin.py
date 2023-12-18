from django.contrib import admin
from .models import HeadSpace, Cluster, Thought
from django import forms

# Register your models here.


class ThoughtAdminForm(forms.ModelForm):
    class Meta:
        model = Thought
        fields = '__all__'
        widgets = {
            'cluster': forms.Select(attrs={'required': False}),
        }


class ThoughtAdmin(admin.ModelAdmin):
    form = ThoughtAdminForm


admin.site.register(HeadSpace)
admin.site.register(Cluster)
admin.site.register(Thought, ThoughtAdmin)
