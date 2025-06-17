import requests
import json
from datetime import datetime, timedelta
import gradio as gr
import pandas as pd
import traceback
import plotly.express as px
import plotly.graph_objects as go
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NasaSsdCneosApi:
    def __init__(self):
        self.fireball_url = "https://ssd-api.jpl.nasa.gov/fireball.api"
        self.ca_url = "https://ssd-api.jpl.nasa.gov/cad.api"
        
        # For debugging - print response details if True
        self.debug_mode = True
    
    def _make_api_request(self, url, params, name="API"):
        """Generic API request handler with error handling and debugging"""
        try:
            # Clean up None values and empty strings
            clean_params = {k: v for k, v in params.items() if v is not None and v != ""}
            
            # Log the request in debug mode
            if self.debug_mode:
                logger.info(f"{name} Request - URL: {url}")
                logger.info(f"{name} Request - Params: {clean_params}")
            
            # Make the request
            response = requests.get(url, params=clean_params)
            
            # Log the response status and content in debug mode
            if self.debug_mode:
                logger.info(f"{name} Response - Status: {response.status_code}")
                logger.info(f"{name} Response - Content Preview: {response.text[:500]}...")
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            # Check for API-specific error messages
            if isinstance(data, dict) and "error" in data:
                logger.error(f"{name} API Error: {data['error']}")
                return None
                
            return data
            
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"{name} HTTP Error: {http_err}")
            if self.debug_mode and hasattr(http_err, 'response'):
                logger.error(f"Response content: {http_err.response.text}")
            return None
            
        except json.JSONDecodeError as json_err:
            logger.error(f"{name} JSON Decode Error: {json_err}")
            if self.debug_mode and 'response' in locals():
                logger.error(f"Raw response: {response.text}")
            return None
            
        except Exception as e:
            logger.error(f"{name} General Error: {e}")
            traceback.print_exc()
            return None

    def get_fireballs(self, limit=10, date_min=None, energy_min=None):
        """Get fireball events from NASA CNEOS API"""
        params = {'limit': limit}
        if date_min:
            params['date-min'] = date_min
        if energy_min:
            params['energy-min'] = energy_min
            
        return self._make_api_request(self.fireball_url, params, "Fireball API")

    def get_close_approaches(self, dist_max=None, date_min=None, date_max=None,
                             h_min=None, h_max=None, v_inf_min=None, v_inf_max=None,
                             limit=10):
        """Get close approach data from NASA CNEOS API"""
        params = {
            'limit': limit, 
            'dist-max': dist_max, 
            'date-min': date_min,
            'date-max': date_max, 
            'h-min': h_min, 
            'h-max': h_max,
            'v-inf-min': v_inf_min, 
            'v-inf-max': v_inf_max, 
            'sort': 'date'
        }
            
        return self._make_api_request(self.ca_url, params, "Close Approaches API")

    def format_response(self, data, format_type):
        """Format JSON response from API into a pandas DataFrame"""
        try:
            if not data:
                logger.warning(f"No data received for {format_type} format")
                return None

            # Some API responses use 'signature' field instead of 'fields'
            fields = data.get('fields', data.get('signature'))
            rows = data.get('data')

            if not fields or not rows:
                logger.warning(f"Missing fields or data rows for {format_type} format")
                logger.debug(f"Data structure: {data.keys()}")
                return None

            # Create DataFrame from the API response
            df = pd.DataFrame([dict(zip(fields, row)) for row in rows])
            
            if df.empty:
                logger.warning(f"Empty DataFrame created for {format_type}")
                return None
                
            # Log available columns for debugging
            if self.debug_mode:
                logger.info(f"Available columns in {format_type} response: {df.columns.tolist()}")

            # Format based on data type
            if format_type == 'fireballs':
                # Only rename columns that exist in the DataFrame
                rename_map = {
                    'date': 'Date/Time', 
                    'energy': 'Energy (kt)',
                    'impact-e': 'Impact Energy (10^10 J)', 
                    'lat': 'Latitude',
                    'lon': 'Longitude', 
                    'alt': 'Altitude (km)',
                    'vel': 'Velocity (km/s)'
                }
                # Filter rename map to only include columns that exist
                valid_rename = {k: v for k, v in rename_map.items() if k in df.columns}
                return df.rename(columns=valid_rename)

            elif format_type == 'close_approaches':
                rename_map = {
                    'des': 'Object', 
                    'orbit_id': 'Orbit ID', 
                    'cd': 'Time (TDB)',
                    'dist': 'Nominal Distance (au)', 
                    'dist_min': 'Minimum Distance (au)',
                    'dist_max': 'Maximum Distance (au)', 
                    'v_rel': 'Velocity (km/s)',
                    'h': 'H (mag)'
                }
                valid_rename = {k: v for k, v in rename_map.items() if k in df.columns}
                return df.rename(columns=valid_rename)

            return df
            
        except Exception as e:
            logger.error(f"Data formatting error for {format_type}: {e}")
            traceback.print_exc()
            return None


# Gradio Interface Functions with better error handling

def fetch_fireballs(limit, date_min, energy_min):
    """Fetch fireball data for Gradio interface"""
    try:
        api = NasaSsdCneosApi()
        
        # Process inputs 
        date_min = date_min.strip() if date_min else None
        try:
            energy_min = float(energy_min) if energy_min else None
        except ValueError:
            return f"Error: Invalid energy value '{energy_min}'. Please enter a valid number.", None
        
        data = api.get_fireballs(
            limit=int(limit),
            date_min=date_min,
            energy_min=energy_min
        )
        
        if not data:
            return "No data returned from API. There might be an issue with the connection or parameters.", None
        
        df = api.format_response(data, 'fireballs')
        if df is None or df.empty:
            return "No fireball data available for the specified parameters.", None
        
        # Create world map of fireballs
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            try:
                # Create size column if Energy (kt) is not available
                size_col = 'Energy (kt)' if 'Energy (kt)' in df.columns else None
                
                fig = px.scatter_geo(df, 
                                   lat='Latitude', 
                                   lon='Longitude',
                                   size=size_col,
                                   hover_name='Date/Time' if 'Date/Time' in df.columns else None,
                                   projection='natural earth',
                                   title='Fireball Events')
                
                return df, fig
            except Exception as plot_err:
                logger.error(f"Error creating fireball plot: {plot_err}")
                return df, None
        
        return df, None
    except Exception as e:
        logger.error(f"Error in fetch_fireballs: {e}")
        traceback.print_exc()
        return f"An error occurred: {str(e)}", None

def fetch_close_approaches(limit, dist_max, date_min, date_max, h_min, h_max, v_inf_min, v_inf_max):
    """Fetch close approach data for Gradio interface"""
    try:
        api = NasaSsdCneosApi()
        
        # Process inputs with error handling
        try:
            dist_max = float(dist_max) if dist_max else None
            h_min = float(h_min) if h_min else None
            h_max = float(h_max) if h_max else None
            v_inf_min = float(v_inf_min) if v_inf_min else None
            v_inf_max = float(v_inf_max) if v_inf_max else None
        except ValueError as ve:
            return f"Error: Invalid numeric input - {str(ve)}", None
        
        date_min = date_min.strip() if date_min else None
        date_max = date_max.strip() if date_max else None
        
        data = api.get_close_approaches(
            limit=int(limit),
            dist_max=dist_max,
            date_min=date_min,
            date_max=date_max,
            h_min=h_min,
            h_max=h_max,
            v_inf_min=v_inf_min,
            v_inf_max=v_inf_max
        )
        
        if not data:
            return "No data returned from API. There might be an issue with the connection or parameters.", None
        
        df = api.format_response(data, 'close_approaches')
        if df is None or df.empty:
            return "No close approach data available for the specified parameters.", None
        
        # Create scatter plot
        try:
            x_col = 'Nominal Distance (au)' if 'Nominal Distance (au)' in df.columns else df.columns[0]
            y_col = 'Velocity (km/s)' if 'Velocity (km/s)' in df.columns else df.columns[1]
            hover_col = 'Object' if 'Object' in df.columns else None
            size_col = 'H (mag)' if 'H (mag)' in df.columns else None
            color_col = 'H (mag)' if 'H (mag)' in df.columns else None
            
            fig = px.scatter(df, 
                          x=x_col, 
                          y=y_col,
                          hover_name=hover_col,
                          size=size_col,
                          color=color_col,
                          title='Close Approaches - Distance vs Velocity')
            
            return df, fig
        except Exception as plot_err:
            logger.error(f"Error creating close approach plot: {plot_err}")
            return df, None
    except Exception as e:
        logger.error(f"Error in fetch_close_approaches: {e}")
        traceback.print_exc()
        return f"An error occurred: {str(e)}", None

# Create Gradio interface
with gr.Blocks(title="NASA SSD/CNEOS API Explorer") as demo:
    gr.Markdown("# NASA SSD/CNEOS API Explorer")
    gr.Markdown("Access data from NASA's Center for Near Earth Object Studies")
    
    # Error display area
    error_box = gr.Textbox(label="Status", visible=True)
    
    with gr.Tab("Fireballs"):
        gr.Markdown("### Fireball Events")
        gr.Markdown("Get information about recent fireball events detected by sensors.")
        with gr.Row():
            with gr.Column():
                fireball_limit = gr.Slider(minimum=1, maximum=100, value=10, step=1, label="Limit")
                fireball_date = gr.Textbox(label="Minimum Date (YYYY-MM-DD)", placeholder="e.g. 2023-01-01")
                fireball_energy = gr.Textbox(label="Minimum Energy (kt)", placeholder="e.g. 0.5")
                fireball_submit = gr.Button("Fetch Fireballs")
            with gr.Column():
                fireball_results = gr.DataFrame(label="Fireball Results")
                fireball_map = gr.Plot(label="Fireball Map")
        
        fireball_submit.click(fetch_fireballs, 
                            inputs=[fireball_limit, fireball_date, fireball_energy], 
                            outputs=[fireball_results, fireball_map])
    
    with gr.Tab("Close Approaches"):
        gr.Markdown("### Close Approaches")
        gr.Markdown("Get information about close approaches of near-Earth objects.")
        with gr.Row():
            with gr.Column():
                ca_limit = gr.Slider(minimum=1, maximum=100, value=10, step=1, label="Limit")
                ca_dist_max = gr.Textbox(label="Maximum Distance (AU)", placeholder="e.g. 0.05")
                ca_date_min = gr.Textbox(label="Minimum Date (YYYY-MM-DD)", placeholder="e.g. 2023-01-01")
                ca_date_max = gr.Textbox(label="Maximum Date (YYYY-MM-DD)", placeholder="e.g. 2023-12-31")
                ca_h_min = gr.Textbox(label="Minimum H (mag)", placeholder="e.g. 20")
                ca_h_max = gr.Textbox(label="Maximum H (mag)", placeholder="e.g. 30")
                ca_v_min = gr.Textbox(label="Minimum Velocity (km/s)", placeholder="e.g. 10")
                ca_v_max = gr.Textbox(label="Maximum Velocity (km/s)", placeholder="e.g. 30")
                ca_submit = gr.Button("Fetch Close Approaches")
            with gr.Column():
                ca_results = gr.DataFrame(label="Close Approach Results")
                ca_plot = gr.Plot(label="Close Approach Plot")
        
        ca_submit.click(fetch_close_approaches, 
                      inputs=[ca_limit, ca_dist_max, ca_date_min, ca_date_max, ca_h_min, ca_h_max, ca_v_min, ca_v_max], 
                      outputs=[ca_results, ca_plot])
    
    gr.Markdown("### About")
    gr.Markdown("""
    This application provides access to NASA's Solar System Dynamics (SSD) and Center for Near Earth Object Studies (CNEOS) API.
    
    Data is retrieved in real-time from NASA's servers. All data is courtesy of NASA/JPL-Caltech.
    
    Created using Gradio and Hugging Face Spaces.
    """)

# Create requirements.txt file
requirements = """
gradio>=3.50.0
pandas>=1.5.0
plotly>=5.14.0
requests>=2.28.0
"""

with open("requirements.txt", "w") as f:
    f.write(requirements)

if __name__ == "__main__":
    demo.launch()
