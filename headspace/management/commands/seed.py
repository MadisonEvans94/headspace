from django.core.management.base import BaseCommand
from headspace.models import HeadSpace, Cluster, Thought
import random

USERS = ["MADISON"]


THOUGHT_CONTENT = [
    "Deep Learning is an interesting field",
    "I see a lot of promise with the future of AI",
    "Music is my creative outlet",
    "Musical Concerts have changed a lot in the past year"
]

CLUSTERS = [
    "ARTIFICIAL INTELLIGENCE",
    "MUSIC"
]

# Global parameters
NUM_HEADSPACES = len(USERS)
NUM_CLUSTERS = len(CLUSTERS)
NUM_THOUGHTS = len(THOUGHT_CONTENT)
EMBEDDING_VECTOR_LENGTH = 4


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
                settings={"setting1": "on", "setting2": "off"},
                name=f"{USERS[_]}'s Headspace",)

        # Create Clusters
        for _ in range(NUM_CLUSTERS):
            Cluster.objects.create(
                name=CLUSTERS[_],
                description=f"Description for {CLUSTERS[_]} Cluster ",
                embedding_vector=[random.uniform(
                    0.0, 1.0) for _ in range(EMBEDDING_VECTOR_LENGTH)]
            )

        # Create Thoughts
        headspaces = HeadSpace.objects.all()
        clusters = Cluster.objects.all()
        for content in THOUGHT_CONTENT:
            Thought.objects.create(
                headspace=random.choice(headspaces),
                cluster=random.choice(clusters),
                content=content,
                embedding_vector=[random.uniform(
                    0.0, 1.0) for _ in range(EMBEDDING_VECTOR_LENGTH)]
            )

        self.stdout.write(self.style.SUCCESS(
            'Successfully seeded the database'))
