import falcon

from .ventilator_prediction_route import ventilator_prediction_route


api = application = falcon.API()

vent = ventilator_prediction_route()
api.add_route("/vent", vent)
