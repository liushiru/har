window = [None, None, None]

prediction = [None, None, None]

def predict():
    pass


import time

postion_pred = [None, None, None]
dance_pred = [None, None, None]
position_ready = False
dance_ready = False
position_send = None
dance_send = None

receive_time = time.time()


def receive():
    position_ready = False
    dance_ready = False
    # clear Queue
    receive_time = time.time()
    postion_pred = [None, None, None]
    dance_pred = [None, None, None]


def do_ml(dancer_id):
    prediction = ml(window)
    if prediction >= 4 and (position_pred[dancer_id] == 6 or position_pred[dancer_id] == None):
        position_pred[dancer_id] = prediction
    elif prediction < 4:
        dance_pred[dancer_id] = prediction


def determine_dance_move(dance_pred):
    dance_pred = [2 if pred is None else pred for pred in dance_pred]
    counter, freq = Counter(dance_pred).most_common[1][0]
    if freq > 2:
        return counter
    else:
        return None


self.receive_time = time.time()
def manage_prediction():
    while True:
        try:
            if position_ready == False and (time.time() - receive_time > 2 or None not in position_pred):
                position_send = determine_position(curr_dance_pos, position_pred)
                position_ready = True
            else:
                continue
            if time.time() - receive_time > 8:
                dance_send = 1
                dance_ready = True

            if dance_ready == False and (None not in position_pred):
                dance_pred = determine_dance_move(dance_pred)
                dance_send = dance_pred
                if dance_send is not None:
                    dance_ready = True








            # No dc case
            if None not in self.prediction_dancers and self.prediction_dancers == False:
                counter, freq = Counter(self.prediction_dancers).most_common[1][0]
                if freq >= 2:
                    self.dance_move_prediction = counter
                    self.prediction_ready = True



        except:
