
from base64 import b64encode
import requests


def get_token():
  url = 'https://accounts.spotify.com/api/token'
  client_id = 'xxx'
  client_secret = 'xxx'
  form_data = {
    'grant_type': 'client_credentials',
  }
  return requests.post(url, auth=(client_id, client_secret), data=form_data).json().get('access_token')


def get_nested_item(data, path):
  keys = path.split('.')
  for key in keys:
    data = data.get(key, {})
  return data


def poll_paginated_url(token, url, type, item_path, total_path):
  headers = {
    "Authorization": f'Bearer {token}'
  }

  offset = 0
  total = 1
  results = []

  print(url)

  while offset < total:
    full_url = url + f'?offset={offset}&limit={50}'
    result = requests.get(full_url, headers=headers).json()
    results += get_nested_item(result, item_path)
    total = get_nested_item(result, total_path)
    offset += 50
    print(f'read {offset} of {total} {type}')

  return results


def get_featured_playlists(token):
  url = 'https://api.spotify.com/v1/browse/featured-playlists'

  playlists = poll_paginated_url(token, url, 'playlists', 'playlists.items', 'playlists.total')

  return set(playlist['id'] for playlist in playlists)


def get_playlist_artists(token, playlist_id):
  url = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks'

  tracks = poll_paginated_url(token, url, 'artists', 'items', 'total')

  artists = [
      artist
      for track in tracks
      if track.get('track') and track['track'].get('artists')
      for artist in track['track']['artists']
  ]

  return set(artist['name'] for artist in artists)


# If I want to get more names I can extend this to grab more playlists
token = get_token()
playlists = get_featured_playlists(token)

# My Danger Playlist
playlists.add('xxx')

artists = set()

for idx, playlist in enumerate(playlists):
  print(f'getting artists in playlist {idx + 1} of {len(playlists)}')
  artists |= get_playlist_artists(token, playlist)

with open('./artists.txt', 'w', encoding='utf-8') as outfile:
  outfile.write('\n'.join(str(i) for i in artists))
