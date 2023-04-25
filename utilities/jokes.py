import random
class Jokes:
    def __init__(self):
        self.jokes = []
        self.jokes.append('Why did Adele cross the road? To say hello from the other side.')
        self.jokes.append('To the guy who invented zero, thanks for nothing.')
        self.jokes.append('Ladies, if he can’t appreciate your fruit jokes, you need to let that mango.')
        self.jokes.append('I don’t trust stairs because they’re always up to something.')

    def return_jokes(self, index):
        return self.jokes[index]
