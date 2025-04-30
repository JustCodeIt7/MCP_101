from mcp.server.fastmcp import FastMCP
import httpx
import asyncio

# Instantiate FastMCP
mcp = FastMCP("weather_forecast")

@mcp.tool()
async def get_forecast(lat: float, lon: float) -> str:
    """
    Get weather forecast for the given latitude and longitude.
    
    Args:
        lat: Latitude (decimal degrees)
        lon: Longitude (decimal degrees)
        
    Returns:
        A string containing the forecast for the next few periods
    """
    try:
        async with httpx.AsyncClient() as client:
            # First, get the forecast URL from the points endpoint
            points_url = f"https://api.weather.gov/points/{lat},{lon}"
            response = await client.get(points_url)
            
            if response.status_code != 200:
                return "No forecast available: Could not get forecast URL"
            
            # Extract the forecast URL from the response
            try:
                forecast_url = response.json()["properties"]["forecast"]
            except (KeyError, ValueError):
                return "No forecast available: Invalid response from weather service"
            
            # Now get the actual forecast
            forecast_response = await client.get(forecast_url)
            
            if forecast_response.status_code != 200:
                return "No forecast available: Could not retrieve forecast"
            
            try:
                forecast_data = forecast_response.json()
                periods = forecast_data["properties"]["periods"]
                
                # Format the first 3 periods into a readable string
                result = []
                for i, period in enumerate(periods[:3]):
                    result.append(f"{period['name']}: {period['temperature']}°{period['temperatureUnit']} - {period['shortForecast']}")
                
                return "\n".join(result)
            except (KeyError, ValueError, IndexError):
                return "No forecast available: Could not parse forecast data"
    except Exception as e:
        return f"No forecast available: Error occurred - {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")