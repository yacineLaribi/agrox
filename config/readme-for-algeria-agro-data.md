# Algeria Agro Scraper (Developer Overview)

This Node.js script collects **weather and soil data** for all Algerian wilayas using the Agro Monitoring API and stores the results in a JSON file.

---

## What this script does

For each wilaya in Algeria:

1. Uses its latitude and longitude
2. Creates a temporary polygon (small farmland area) via the Agro Monitoring API
3. Calls two APIs:
   - Weather API (air temperature, air humidity)
   - Soil API (moisture + soil temperature)
4. If soil data is not ready, it retries automatically (polling)
5. Deletes the polygon to avoid API clutter
6. Saves the final data into a JSON file

---

## APIs used

This script uses **Agro Monitoring API** endpoints:

- Polygon creation API
- Soil data API (requires polygon)
- Weather data API (uses coordinates)

Official API documentation:
https://agromonitoring.com/api

---

## How to run it

### 1. Install dependencies

```bash
npm install axios dotenv
```

### 2. Create a .env file

```env
AGRO_API_KEY=your_api_key_here
```

### 3. Run the scraper

```bash
node full_algeria_scraper.js
```

### 4. Output file

The final data is written to:

```
algeria_agro_data.json
```

---

## How it works internally

### Polygon logic
The script draws a small square polygon around each wilaya’s coordinates (about 15 hectares).  
This polygon is required by the soil API.

### Polling system
The soil API sometimes returns 404 or 400 initially.  
The script:
- Retries every 30 seconds
- Stops after 10 attempts

### Weather logic
Weather data is fetched instantly using coordinates (no polygon required).

### Cleanup logic
Every polygon is deleted after data is collected to avoid API quota issues.

---

## Configuration variables

Located inside the script:

```js
POLLING_INTERVAL_MS = 30000;
MAX_POLLING_ATTEMPTS = 10;
RATE_LIMIT_DELAY_MS = 500;
```

---

## Notes for developers

- Soil data depends on polygon creation
- Do not run multiple instances in parallel (rate-limit risk)
- API free tier has request limits
- Duplicate polygon errors are handled (HTTP 409)
- Network errors are caught and logged

---

## Summary for devs

This is basically:
- A batch processor
- That creates polygons
- Pulls weather + soil data
- Handles missing data with retries
- Cleans up after itself
- Saves results to JSON
