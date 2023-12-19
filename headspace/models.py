from django.db import models
from django.core.exceptions import ValidationError

EMBEDDING_VECTOR_LENGTH = 32

# Validators


def validate_embedding_vector(value):
    if len(value) != EMBEDDING_VECTOR_LENGTH:
        raise ValidationError(
            f"The list must be exactly {EMBEDDING_VECTOR_LENGTH} elements long.")

    if not all(isinstance(element, float) for element in value):
        raise ValidationError("All elements of the list must be floats.")

# Models


class HeadSpace(models.Model):
    name = models.TextField(max_length=100, null=True)
    settings = models.JSONField()

    def __str__(self):
        return self.name


class Cluster(models.Model):
    name = models.TextField(max_length=100)
    embedding_vector = models.JSONField(
        validators=[validate_embedding_vector])
    description = models.TextField(max_length=1000)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True, null=True)

    def __str__(self):
        return self.name


class Thought(models.Model):
    headspace = models.ForeignKey(HeadSpace, on_delete=models.CASCADE)
    cluster = models.ForeignKey(
        Cluster, on_delete=models.SET_NULL, null=True)
    content = models.TextField(max_length=2000)
    created_at = models.DateTimeField(auto_now_add=True)
    embedding_vector = models.JSONField(
        validators=[validate_embedding_vector])

    def __str__(self):
        return f"Thought ID: {self.id}"
