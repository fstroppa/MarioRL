# MarioRL

A simple algorithm that makes Mario beat some levels. Some of my thoughts during the project:

**1 - Why use a simple algorithm instead of the overall Deep RL or meta-heuristics?**
  It is not needed. I intended to make Mario beat some levels with the minimum information possible. The only input is the time and if Mario dies, wins or gets stuck.
  In a project to optimize a water network, we would need to have the minimum valves and sensors installed as possible. So it would be great to not rely on unnecessary information.

**2 - When to use a prototype and when to use the 'tracer-bullet' idea?**
  Prootypes are usually good when you want to test an idea and don't need it to be scalable. This is an example of this project. If I needed to scale it, I could restart the code from the beginning by doing proper software development.

**3 - Is Domain Driven Design still useful?**
  A lot of the design patterns are useful, we need to know when to apply them. For example, it would not have made any sense to use TDD here (it is a prototype, I didn't even do unit tests). The knowledge of the domain allowed me to implement a few behaviors into the model (for example, that it should go right).
  In an airline company, for example, we can use past information for some routes or scheduling instead of building something from zero. Building deep neural networks that know nothing about the problem can make us over-engineer the program.

  **4 - Can we call this 'AI'? It only reads the frame in which the game is on!**
    A model is always a simplification of what is happening in the real world. A good engineering process allows us to get the minimum relevant information to construct the model and to meet our goal.

  **5 - So if there is no reward in that algorithm, can we call it Reinforcement Learning?**
    I don't know if we can call it RL, but the goal was to beat some levels, and it does it well! Training time takes like 2 minutes.
