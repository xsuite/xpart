import xpart as xp
import xtrack as xt

p = xp.Particles(x=[0, 1, 2, 3, 4, 5, 6], _capacity=10)
p.state[[0,3,4]] = 0

tracker = xt.Tracker(line=xt.Line())
tracker.track(p)

p.sort()

#p.reorganize()