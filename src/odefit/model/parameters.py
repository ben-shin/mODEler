from odefit.model.reaction import Reaction


def get_parameters_from_reactions(
    reactions: list[Reaction],
) -> list[str]:
    """
    Get ordered parameter names from parsed reactions.

    Rules:
    - Every reaction contributes its forward rate.
    - Reversible reactions contribute their reverse rate.
    - Irreversible reactions do not contribute a reverse rate.
    - Duplicate parameter names are ignored while preserving order.
    """

    parameters: list[str] = []

    for reaction in reactions:
        if reaction.forward_rate is None:
            raise ValueError("Reaction is missing forward rate")

        if reaction.forward_rate not in parameters:
            parameters.append(reaction.forward_rate)

        if reaction.reversible:
            if reaction.reverse_rate is None:
                raise ValueError("Reversible reaction is missing reverse rate")

            if reaction.reverse_rate not in parameters:
                parameters.append(reaction.reverse_rate)

    return parameters


def detect_parameters(
    reactions: list[Reaction],
) -> list[str]:
    """
    Backward-compatible alias for get_parameters_from_reactions.
    """

    return get_parameters_from_reactions(reactions)


def get_parameter_names_from_reactions(
    reactions: list[Reaction],
) -> list[str]:
    """
    Backward-compatible alias for get_parameters_from_reactions.
    """

    return get_parameters_from_reactions(reactions)


def get_parameters(
    reactions: list[Reaction],
) -> list[str]:
    """
    Backward-compatible alias for get_parameters_from_reactions.
    """

    return get_parameters_from_reactions(reactions)
