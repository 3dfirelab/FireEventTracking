import os
import requests
import pdb 


def file_exists(url):
    """
    Check if Geofabrik .osm.pbf file exists, correctly handling 302 redirects.
    Returns True only if the target or redirect URL ends with '.osm.pbf'.
    """
    try:
        r = requests.head(url, allow_redirects=False, timeout=10)
    except requests.RequestException:
        return False

    if r.status_code == 200:
        # direct hit
        return True
    elif r.status_code in (301, 302, 303, 307, 308):
        loc = r.headers.get('Location', '')
        # Check if redirect leads to a .osm.pbf file
        if loc.endswith(".osm.pbf"):
            return True
        # If redirect is just to the homepage or blank, it’s not found
        if loc.strip('/') in ['', 'https://download.geofabrik.de']:
            return False
        # In ambiguous cases, follow once to confirm
        try:
            follow = requests.head(loc, allow_redirects=True, timeout=10)
            return follow.status_code == 200 and follow.url.endswith(".osm.pbf")
        except requests.RequestException:
            return False
    else:
        return False

dirData = '/data/shared/OSM/'

# Mediterranean countries list
med = [
    ['Tunisia', 'TUN', None],
    ['Syria', 'SYR', None],
    ['Lebanon', 'LBN', None],
    ['Jordan', 'JOR', None],
    ['Israel and Palestine', 'IS1', None],
    ['Egypt', 'EGY', None],
    ['Morocco', 'MAR,SAH', None],
    ['Algeria', 'DZA', None],
    ['Libya', 'LBY', None],
    ['Albania', 'AL', None],
    ['Andorra', None, None],
    ['Azores', 'PT', 'PT2'],
    ['Austria', 'AT', None],
    ['Belgium', 'BE', None],
    ['Bosnia-Herzegovina', None, None],
    ['Bulgaria', 'BG', None],
    ['Croatia', 'HR', None],
    ['Cyprus', 'CY', None],
    ['Czech Republic', 'CZ', None],
    ['Faroe Islands', None, None],
    ['Finland', 'FI', None],
    ['France', 'FR', '!=FRY'],
    ['Germany', 'DE', None],
    ['Great Britain', 'UK', None],
    ['Greece', 'EL', None],
    ['guernsey-jersey', None, None],
    ['Hungary', 'HU', None],
    ['Italy', 'IT', None],
    ['Kosovo', None, None],
    ['Latvia', 'LV', None],
    ['Liechtenstein', 'LI', None],
    ['Lithuania', 'LT', None],
    ['Luxembourg', 'LU', None],
    ['Macedonia', 'MK', None],
    ['Malta', 'MT', None],
    ['Monaco', None, None],
    ['Montenegro', None, None],
    ['Netherlands', 'NL', None],
    ['Portugal', 'PT', 'PT1'],
    ['Romania', 'RO', None],
    ['Serbia', None, None],
    ['Slovakia', 'SK', None],
    ['Slovenia', 'SI', None],
    ['Spain', 'ES', None],
    ['Switzerland', 'CH', None],
    ['Turkey', 'TR', None],
]

# Potential regions on the Geofabrik server
continents = ["europe", "asia", "africa"]

# Directory to save files
os.makedirs("osm_data", exist_ok=True)

missingCountries = []
for country_entry in med:
    country_name = country_entry[0]
    country_slug = country_name.lower().replace(" ", "-").replace("’", "")
    
    success = False
    for cont in continents:
        url = f"https://download.geofabrik.de/{cont}/{country_slug}-latest.osm.pbf"
        print(f"Trying {url}")
        r = requests.head(url)
        if file_exists(url):
            print(f"Downloading {country_name} from {cont}...")
            response = requests.get(url, stream=True,)
            out_path = f"{dirData}/{response.url.split('/')[-1]}"
            with open(out_path, "wb") as f:
                for chunk in response.iter_content(1024 * 1024):
                    f.write(chunk)
            print(f"Saved: {out_path}\n")
            success = True
            break
    
    if not success:
        print(f"⚠️ Could not find file for {country_name}\n")
        missingCountries.append(country_entry)
    
