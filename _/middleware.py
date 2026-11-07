
from django.http import JsonResponse
from django.conf import settings

class MaintenanceModeMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        value = getattr(settings, "MAINTENANCE_MODE", "FALSE")

        if str(value).upper() == "TRUE":
            json_message= {
                'status': "error",
                'message': "System is currently in maintenance mode. Please try again later.",
                "status_code": 503
            }
        
            return JsonResponse(json_message, status=503)
        return self.get_response(request)