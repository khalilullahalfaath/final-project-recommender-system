"""
Mock data for music recommendations
"""


def _cover(song_id: int) -> str:
    """Build a deterministic cover URL using picsum.photos seed.

    Same seed always returns the same image, so each song gets a stable cover.
    Resolution 300x300 keeps cards crisp at typical column widths (~180-220px).
    """
    return f"https://picsum.photos/seed/spotipai{song_id}/300/300"


# Hardcoded mock songs data
MOCK_SONGS = [
    {"id": 1, "title": "Midnight Dreams", "artist": "The Echoes", "cover_url": _cover(1), "genre": "Pop"},
    {"id": 2, "title": "Electric Sunset", "artist": "Neon Waves", "cover_url": _cover(2), "genre": "Electronic"},
    {"id": 3, "title": "Acoustic Soul", "artist": "River Stone", "cover_url": _cover(3), "genre": "Acoustic"},
    {"id": 4, "title": "Urban Rhythm", "artist": "City Beats", "cover_url": _cover(4), "genre": "Hip Hop"},
    {"id": 5, "title": "Jazz Nights", "artist": "Smooth Trio", "cover_url": _cover(5), "genre": "Jazz"},
    {"id": 6, "title": "Rock Anthem", "artist": "Thunder Road", "cover_url": _cover(6), "genre": "Rock"},
    {"id": 7, "title": "Classical Morning", "artist": "Symphony Orchestra", "cover_url": _cover(7), "genre": "Classical"},
    {"id": 8, "title": "Indie Vibes", "artist": "The Wanderers", "cover_url": _cover(8), "genre": "Indie"},
    {"id": 9, "title": "Country Roads", "artist": "Nashville Stars", "cover_url": _cover(9), "genre": "Country"},
    {"id": 10, "title": "R&B Groove", "artist": "Soul Collective", "cover_url": _cover(10), "genre": "R&B"},
    {"id": 11, "title": "Latin Fire", "artist": "Salsa Kings", "cover_url": _cover(11), "genre": "Latin"},
    {"id": 12, "title": "Blues Highway", "artist": "Delta Blues Band", "cover_url": _cover(12), "genre": "Blues"},
    {"id": 13, "title": "Reggae Sunshine", "artist": "Island Rhythms", "cover_url": _cover(13), "genre": "Reggae"},
    {"id": 14, "title": "Metal Storm", "artist": "Iron Legion", "cover_url": _cover(14), "genre": "Metal"},
    {"id": 15, "title": "Folk Tales", "artist": "Mountain Echo", "cover_url": _cover(15), "genre": "Folk"},
    {"id": 16, "title": "Dance Floor", "artist": "DJ Pulse", "cover_url": _cover(16), "genre": "Dance"},
    {"id": 17, "title": "Ambient Space", "artist": "Cosmic Sounds", "cover_url": _cover(17), "genre": "Ambient"},
    {"id": 18, "title": "Punk Energy", "artist": "Rebel Youth", "cover_url": _cover(18), "genre": "Punk"},
    {"id": 19, "title": "Soul Searching", "artist": "Deep Voices", "cover_url": _cover(19), "genre": "Soul"},
    {"id": 20, "title": "Disco Fever", "artist": "Groove Machine", "cover_url": _cover(20), "genre": "Disco"},
    {"id": 21, "title": "Synthwave Dreams", "artist": "Retro Future", "cover_url": _cover(21), "genre": "Synthwave"},
    {"id": 22, "title": "Gospel Praise", "artist": "Heavenly Choir", "cover_url": _cover(22), "genre": "Gospel"},
    {"id": 23, "title": "World Beat", "artist": "Global Fusion", "cover_url": _cover(23), "genre": "World"},
    {"id": 24, "title": "Trap Nation", "artist": "Bass Drops", "cover_url": _cover(24), "genre": "Trap"},
    {"id": 25, "title": "Lofi Chill", "artist": "Study Beats", "cover_url": _cover(25), "genre": "Lofi"},
    {"id": 26, "title": "K-Pop Star", "artist": "Seoul Sensation", "cover_url": _cover(26), "genre": "K-Pop"},
    {"id": 27, "title": "House Party", "artist": "Club Masters", "cover_url": _cover(27), "genre": "House"},
    {"id": 28, "title": "Techno Pulse", "artist": "Underground Beats", "cover_url": _cover(28), "genre": "Techno"},
    {"id": 29, "title": "Alternative Edge", "artist": "Modern Sound", "cover_url": _cover(29), "genre": "Alternative"},
    {"id": 30, "title": "Experimental Noise", "artist": "Avant Garde", "cover_url": _cover(30), "genre": "Experimental"},
]


def get_collaborative_recommendations(username):
    """
    Get collaborative filtering recommendations for a user.

    Args:
        username: Username to get recommendations for

    Returns:
        dict: Dictionary with three lists of songs for different recommendation categories
    """
    return {
        "users_like_you": MOCK_SONGS[0:10],
        "popular_similar": MOCK_SONGS[10:20],
        "trending_network": MOCK_SONGS[20:30],
    }


def get_content_based_recommendations(username):
    """
    Get content-based recommendations for a user.

    Args:
        username: Username to get recommendations for

    Returns:
        dict: Dictionary with three lists of songs for different recommendation categories
    """
    return {
        "similar_favorites": MOCK_SONGS[5:15],
        "more_like_this": MOCK_SONGS[15:25],
        "genre_based": MOCK_SONGS[0:10],
    }
