class Newton: 
    def __init__(self, f, fprime, tol = 1e-8, iterations = 300, bounds = None, damp = 0.3):
        self.f = f 
        self.fprime = fprime 
        self.tol = tol
        self.iter = iterations 
        self.bounds = bounds 
        self.damping = damp 

    def solve(self, guess): 
        self.guess = guess 
        for i in range(self.iter):
            if self.bounds: 
                minimum, maximum = self.bounds
                if not (minimum < self.guess < maximum): 
                    raise ValueError("Initial Guess outside bounds")
                
            fx = self.f(guess)
            fprime = self.fprime(guess) 
            step = -self.damping * fx/fprime
            new = guess + step 

            if self.bounds: 
                if new <= minimum or new >= maximum: 
                    new = 0.5 * (guess + minimum) if step < 0 else 0.5 * (guess + maximum)

            if abs(new - guess) < self.tol:
                print(f'The number of iterations were {i+1}')
                return new

            guess = new
