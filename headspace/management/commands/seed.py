from django.core.management.base import BaseCommand
from headspace.models import HeadSpace, Cluster, Thought
import random

# Global parameters
NUM_HEADSPACES = 10
NUM_CLUSTERS = 5
NUM_THOUGHTS = 50
EMBEDDING_VECTOR_LENGTH = 32


class Command(BaseCommand):
    help = 'Seeds the database with initial data'

    def handle(self, *args, **kwargs):
        # Clear existing data (optional)
        HeadSpace.objects.all().delete()
        Cluster.objects.all().delete()
        Thought.objects.all().delete()

        # Create HeadSpaces
        for _ in range(NUM_HEADSPACES):
            # Replace with actual settings
            HeadSpace.objects.create(
                settings={"setting1": "value1", "setting2": "value2"})

        # Create Clusters
        for _ in range(NUM_CLUSTERS):
            Cluster.objects.create(
                name=f"Cluster {_}",
                description=f"Description for Cluster {_}",
                embedding_vector=[random.uniform(
                    0.0, 1.0) for _ in range(EMBEDDING_VECTOR_LENGTH)]
            )

        # Create Thoughts
        headspaces = HeadSpace.objects.all()
        clusters = Cluster.objects.all()
        for i in range(NUM_THOUGHTS):
            Thought.objects.create(
                headspace=random.choice(headspaces),
                cluster=random.choice(clusters),
                content=f"Content of Thought {i}",
                embedding_vector=[random.uniform(
                    0.0, 1.0) for _ in range(EMBEDDING_VECTOR_LENGTH)]
            )

        self.stdout.write(self.style.SUCCESS(
            'Successfully seeded the database'))
