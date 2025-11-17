# Background on this folder

While solving the pains of better transaction categorization and quick email searching was the primary goal, a secondary goal of this project was to understand today's state of agentic coding. In order to more deeply undestand the current strengths/limitations of coding with LLMs, I leveraged them as much as possible every step of the way.

To name a few tools — AWS Kiro (using Claude 4+), Github Copilot (GPT 4.1+), Kilo Code (Claude 4+, GPT 4.1+), Gemini Pro, and each of the LLM's respective chat interfaces. No matter the tool, I found that hands-off vibe coding (popularized as where you don't review little or none of the LLM's outputs) still doesn't work except for tiny projects/scripts. For projects with medium complexity, LLMs seem to start off strong, but struggle in the last 20% to either fix bugs, or code themselves into a little whack-a-mole mess where fixing one thing breaks another until you're all out of credits. The Pareto principle strikes again and 80% of the time ends up being spent on that last 20% of the project. Without clear, specific, focused instructions on small tasks, I found that I'm spending tons of time reviewing and deleting more code than I'm using.

There's a lot I can write about, but for the sake of brevity, here are the key takeways for leveraging LLMs in medium+ sized projects that became clear throughout the process:
1. Create good interfaces for the entities/services in the app upfront. LLMs tend to struggle in design aspects or anticipating how the system might need to be extended over time. In this folder you will find the Thrift files I created up front that defined the interaction between components. (Yes there's better alternatives to Thrift for intra-system contracts, but I had other reasons for using Thrift)
2. Create layers in the app architecture with clear/distinct responsibilities so that the LLM has good guardrails of where to put what.
3. Write up all the rules of what belongs where, and which things to never do (of course LLMs don't follow these to the dot every time, but it at least pushes them in the right direction)
4. Then take the design, break it into small executable tasks, and execute them one by one while checking the output for correctness piece by piece

I left these .llm artifacts in the repo for those who are curious. Hopefully reviewing the artifacts will teach you something new!

