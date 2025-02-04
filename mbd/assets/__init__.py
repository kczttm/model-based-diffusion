import os

# Get the directory containing the XML files
ASSET_DIR = os.path.dirname(os.path.abspath(__file__))

# Dictionary mapping environment names to their XML files
XML_PATHS = {
    'bouncing_ball': os.path.join(ASSET_DIR, 'bouncing_ball.xml'),
    # Add other XML files here as needed
    'ant': os.path.join(ASSET_DIR, 'ant.xml'),
    'halfcheetah': os.path.join(ASSET_DIR, 'halfcheetah.xml'),
    'hopper': os.path.join(ASSET_DIR, 'hopper.xml'),
    'humanoid': os.path.join(ASSET_DIR, 'humanoid.xml'),
    'walker2d': os.path.join(ASSET_DIR, 'walker2d.xml'),
}

def get_xml_path(env_name):
    """Get the full path to an environment's XML file."""
    if env_name not in XML_PATHS:
        raise ValueError(f"Unknown environment: {env_name}")
    return XML_PATHS[env_name] 