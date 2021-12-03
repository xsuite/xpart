import xpart as xp

# Build a reference particles
p0 = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12, x=1, y=3)

# Built a set of three particles with different x coordinates
particles = xp.build_particles(particle_ref=p0, y=[1,2,3])

# Inspect
print(particles.p0c[1]) # gives 7e12
print(particles.x[1]) # gives 0.0
print(particles.y[1]) # gives 2.0
