import pkgutil
import stacchip

# Print all modules in the stacchip package
print("Modules in stacchip package:")
for module in pkgutil.iter_modules(stacchip.__path__):
    print(f"- {module.name}")

# Also check if there's an older module name
try:
    import stacchip.chipper as chipper
    print("\nstacchip.chipper exists")
except ImportError:
    print("\nstacchip.chipper does not exist")

try:
    import stacchip.chipper_mod as chipper_mod
    print("stacchip.chipper_mod exists")
except ImportError:
    print("stacchip.chipper_mod does not exist")