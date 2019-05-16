"""Main module for App Engine app."""

from flask import Flask, jsonify, request, render_template
from recommendations import Recommendations

app = Flask(__name__)
rec_util = Recommendations()
DEFAULT_RECS = 5

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendation', methods=['GET'])
def recommendation():
    """Given a user id, return a list of recommended movie ids."""
    user_id = request.args.get('userId')
    num_recs = request.args.get('numRecs')

    # validate args
    if user_id is None:
        return 'No User Id provided.', 400
    if num_recs is None:
        num_recs = DEFAULT_RECS
    try:
        uid_int = int(user_id)
        nrecs_int = int(num_recs)
    except:
        return 'User id and number of recs arguments must be integers.', 400

    # Get recommended movies
    rec_list = rec_util.get_recommendations(uid_int, nrecs_int)

    if rec_list is None:
        return 'User Id not found : %s' % user_id, 400

    json_response = jsonify({'movies': [str(i) for i in rec_list]})
    return json_response, 200


@app.route('/prediction', methods=['GET'])
def prediction():
    """Given a user id and a movie id, return a rating """
    user_id = request.args.get('userId')
    movie_id = request.args.get('movieId')

    # validate args
    if user_id is None:
        return 'No User Id provided.', 400
    if movie_id is None:
        return 'No Movie Id provided.', 400
    try:
        uid_int = int(user_id)
        mid_int = int(movie_id)
    except:
        return 'User id and movie id arguments must be integers.', 400

    # Get predicted rating
    rating = rec_util.get_predictions(uid_int, mid_int)

    if rating is None:
        return 'User Id or Movie Id not found : %s' % user_id, 400

    json_response = jsonify({'rating': [str(rating)]})
    return json_response, 200


@app.route('/readiness_check', methods=['GET'])
def readiness_check():
    return '', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
