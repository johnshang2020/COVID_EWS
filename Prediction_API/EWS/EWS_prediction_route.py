import falcon
import json
from .EWS_prediction import EWS_prediction


class EWS_prediction_route(object):
    EWS = EWS_prediction()
    # on get returns a list of 10 random paitent encounter ids
    def on_get(self, req, resp):

        content = self.EWS.get_pat_csn_list()
        resp.body = json.dumps(content, ensure_ascii=False)
        resp.status = falcon.HTTP_200

    # on a post gets a prediction for the specified id
    def on_post(self, req, resp):
        req_body = req.media

        pat_enc_csn_id = int(req_body.get("enc_csn_id"))

        if "num_hours_ahead" in req_body:
            num_hours_ahead = int(req_body.get("num_hours_ahead"))

        valid_enc_csn_id = self.EWS.check_pat_enc_csn_id(pat_enc_csn_id)
        # checks to if the num_hours_head send it outside the window of 12-48 hours and if it is returns a error
        if "num_hours_ahead" in req_body and not 12 <= num_hours_ahead <= 48:
            content = {"error": "num_hours_ahead outside allowed range of 12-48"}
        # if it is in that range send it to prediction
        elif valid_enc_csn_id and "num_hours_ahead " in req_body:
            content = self.EWS.get_prediction(
                n_hours_ahead=num_hours_ahead, s_enc_csn_id=pat_enc_csn_id
            )
        # otherwise if it doesn't exist just uses default of 24 hours
        elif valid_enc_csn_id:
            content = self.EWS.get_prediction(s_enc_csn_id=pat_enc_csn_id)
        else:
            content = {
                "error": "no records found associated with the pat_enc_csn_id: "
                + str(pat_enc_csn_id)
            }

        resp.body = json.dumps(content, ensure_ascii=False)
        resp.status = falcon.HTTP_200

    # checks to make sure parameters are valid, if they are returns true along with the parameters converted into int, otherwise returns false and a error message
    # def check_post_valid(self,post_body):
