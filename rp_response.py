import json

def runpod_response(status_code=200, content_type="application/json", body=None):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": content_type
        },
        "body": json.dumps(body) if body is not None else ""
    }