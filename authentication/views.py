from rest_framework.status import HTTP_200_OK, HTTP_400_BAD_REQUEST
from rest_framework.views import APIView

from django.http import JsonResponse


class Home(APIView):
    def get(self, request):
        return JsonResponse(
            {
                "status": "success",
                "message": "Welcome to ScamShield API",
                "status_code": HTTP_200_OK,
            },
            status=HTTP_200_OK,
        )
