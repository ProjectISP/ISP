# Exceptions Classes
class InvalidFile(Exception):

    def __init__(self, message):
        Exception.__init__(self)
        self.message = message
