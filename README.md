# UniversalNEAT
This is a heavily modified and updated version of the Neuroevolution of Augmenting Topologies (NEAT) algorithm.
The original version couldn't solve complex problems, wasn't universal, and many of its versions didn't support RNNs or many other principles.
This C# program solves all of the above problems and also adds many new, important improvements, thanks to my research. So, it can't really be called a revision—it's a new, important learning algorithm. It can learn with any evaluation function, even non-continuous ones, and is completely universal!
This is a solution to one of the most popular NLP problems: text generation.

# Long, long description about research.

If you are not familiar with the original NEAT, you can read my articles(https://habr.com/ru/articles/910878/, https://habr.com/ru/articles/985454/), the original 2006 paper, or search the internet for a better explanation.

I plan to continue developing this project and describing the new features in more detail. For now, here's a brief overview:

1. **RNN**
Although adding RNN connections was described in the original article, but not in any known implementation. Here, this important mechanism works as follows: we separately calculate the topological sorting of feedforward connections and separately store the RNN list, which we transfer to the neuron state after execution. For complex problems, this is a necessary condition.
It's worth exploring - perhaps the presence of RNNs even in non-sequential tasks helps reduce the size of input neurons, which could be a solution to the problem of large input data.

2. **How to build huge networks for complicated tasks(NLP, generating and analyzing images and so on)?**
This is one of the key innovations, and the solution itself is as simple as the advent of ResNET to combat gradient decay: we make neurons accessible only as they learn, thereby increasing the likelihood of accelerating learning and avoiding problems! Moreover, this is consistent with the evolutionary history. For example, you can study, watch, and read about how the eye evolved.

3. **Training for anything**
Since NEAT doesn't care about continuity, we can use undifferentiated functions directly!

4. **How to improve a best model?**
How can we further train a model obtained after training? In addition to NEAT variations, you can still use standard backward, RL, or AutoML for this. Why not, but NEAT will train better usually. Even without adding neurons as in gradient methods, it still makes sense from an evolutionary perspective—it's how our brain learns after infancy. Neurogenesis occurs only at a very early age; by 4-5 years old, it's definitely gone (if you want to know more, check online). This is why excessive alcohol consumption and concussions are so dangerous—you permanently lose part of your intellect, personality, reason, intelligence, and everything that connected you up to that point. Most importantly, attempts to restore neurogenesis are virtually unknown. We only have neuroplasticity, but it's like the famous chocolate bar experiment—it looks the same, but the volume is definitely smaller, and for it to function properly, you need to observe almost quarantine conditions for 2-3 weeks. Neurons appear only in the hippocampus and one other region, but definitely not in the cerebral cortex. Take very good care of your brain!

5. **How to start generation?**
Direct thatcher forcing can't be used, as it will lead to problems between the RNN state and the actual words it sees, which is unlikely to have a positive effect on learning. It's better to show it separately in a dedicated neuron when we want to get a prediction. And restarting from the beginning each time will be O(n^2). But this can be accelerated to O(n)—see the next point!

5. **How to speed up inference?**
It is possible to save the states of neurons until the last token is submitted with a request to make a prediction, something similar to the optimization of the KV-cache in transformers, so that the RNNs will not have inconsistent data.

**This isn't all there is to it, especially since it's very brief and not full. I'll add more detailed descriptions later. Let me know if anything isn't clear!
Various generative models were used for refactoring, porting to pure C#, and simply assisting with my long-term research. This greatly simplified the development process, but at the very beginning of the research, they weren't available—there weren't any good free models available back then. You can follow the project's progress in my previous NEAT repositories. Good luck!**
