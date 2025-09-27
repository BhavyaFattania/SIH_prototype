import firebase_admin
from firebase_admin import credentials, firestore

from datetime import datetime
import requests
import time
cred = credentials.Certificate("serviceAccountKey.json")  # your Firebase service account JSON
firebase_admin.initialize_app(cred)

db = firestore.client()

def format_datetime(dataTime):
    dt = datetime(
        year=dataTime['year'],
        month=dataTime['monthValue'],
        day=dataTime['dayOfMonth'],
        hour=dataTime.get('hour', 0),
        minute=dataTime.get('minute', 0)
    )
    return dt.strftime("%d-%m-%Y %H:%M")  # 01-06-2025 00:00

districts = [
    #    "Ahmedabad", "Amreli", "Anand", "Aravalli", "Banaskantha", "Bharuch",
    #    "Bhavnagar", "Botad", "Chhota Udaipur", "Dahod", "Dang", "Devbhumi Dwarka",
    #    "Gandhinagar", "Gir Somnath", "Jamnagar", "Junagadh", "Kutch", "Kheda",
    #    "Mahisagar", "Mehsana", "Morbi", "Narmada", "Navsari", "Panchmahal",
       "Patan", "Porbandar", "Rajkot", "Sabarkantha", "Surat", "Surendranagar",
       "Tapi", "Vadodara", "Valsad"
    
    
]

fields_to_store = [
    "stationCode",
    "stationName",
    "state",
    "district",
    "dataValue",
    "unit",
    "wellDepth"
]

startdate = "2025-06-01"
enddate = "2025-06-30"
agency = "CGWB"

for district in districts:
    for page in range(0, 5):  # 0 to 5
        district_url = district.replace(" ", "%20")

        url = (
            f"https://indiawris.gov.in/Dataset/Ground%20Water%20Level?"
            f"stateName=Gujarat&districtName={district_url}"
            f"&agencyName={agency}&startdate={startdate}&enddate={enddate}"
            f"&download=true&page={page}&size=1000"
        )

        response = requests.post(url, headers={"accept": "application/json"})
        print("type of response:", type(response))
        response_data = response.json()
        print("type of response_data:", type(response_data))
        
        # Check if the response returned data
        if response.status_code == 200 and isinstance(response_data, list) and len(response_data) > 0:
            print(f"\n✅ Fetched: {district} (page {page})")
            # print first few records to debug
            print(response_data[:2])

            # Insert each station into Firestore
            for record in response_data:
                filtered_record = {key: record.get(key) for key in fields_to_store}
                formatted_time = format_datetime(record["dataTime"])
                filtered_record["Date_Time"] = formatted_time
                station_code = filtered_record["stationCode"]

        # Reference: Gujarat/{district}/Stations/{stationCode}/Readings/{Date_Time}
                doc_ref = (
                db.collection("Gujarat")
                      .document(district)
                      .collection("Stations")
                      .document(station_code)
                      .collection("Readings")
                      .document(formatted_time)
                )

        # Set the data (this creates a new document each time with timestamp as ID)
                doc_ref.set(filtered_record)

            print(f"✅ All stations for {district} inserted successfully!")
            time.sleep(1)  # be gentle with server

        else:
            print(f"No data to insert for district {district}. Response: {response_data}")
        