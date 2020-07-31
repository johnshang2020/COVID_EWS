import falcon

from .EWS_prediction_route import EWS_prediction_route
api = application = falcon.API(middleware=[AuthMiddleware()])

EWS = EWS_prediction_route()
api.add_route("/EWS", EWS)
