import argparse

def str2cache(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 'ram', '1'):
        return True
    elif v.lower() in ('false', 'disk i/o', '0'):
        return False
    elif v.lower() == 'disk':
        return 'disk'
    else:
        raise argparse.ArgumentTypeError('Expected True, "disk", or False for --cache')
    
def cache_option(value):
    if value == 'RAM':
        return True
    elif value == 'Disk I/O':
        return False
    elif value == 'HDD':
        return 'disk'
    else:
        raise ValueError('Invalid cache option.')