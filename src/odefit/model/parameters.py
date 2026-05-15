from odefit.model.reaction import Reaction


def detect_parameters(reactions: list[Reaction]) -> list[str]:
    """
    Detect kinetic parameters used by a list of reactions

    Forward rates are always included
    Reverse rates are only included for reversible reactions
    """

    parameters: list[str] = []

    for reaction in reactions:
        parameters.append(reaction.forward_rate)

        if reaction.reversible:
            if reaction.reverse_rate is None:
                raise ValueError("Reversible reaction is missing a reverse rate")

            parameters.append(reaction.reverse_rate)

    return parameters
