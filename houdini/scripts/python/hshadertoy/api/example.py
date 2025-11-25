import os
import shadertoy

# basic example of using shadertoy API
# requires API key, get yours from https://www.shadertoy.com/api

api_key = os.environ['SHADERTOY_API_KEY']
app = shadertoy.App(api_key)

# get shader by its unique id
shader = app.get_shader("lsGSDG")

# print some info
print( shader["info"]["name"] )
print( shader["renderpass"][0]["code"] )
