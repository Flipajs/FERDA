from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import range
import json
import urllib.request, urllib.parse, urllib.error

def translate(text, src = '', to = 'en'):
  parameters = ({'langpair': '{0}|{1}'.format(src, to), 'v': '1.0' })
  translated = ''

  for text in (text[index:index + 4500] for index in range(0, len(text), 4500)):
    parameters['q'] = text
    response = json.loads(urllib.request.urlopen('http://ajax.googleapis.com/ajax/services/language/translate', data = urllib.parse.urlencode(parameters).encode('utf-8')).read().decode('utf-8'))

    try:
      translated += response['responseData']['translatedText']
    except:
      pass

  return translated


if __name__ == '__main__':
    translate('Ahoj svete', 'cs')