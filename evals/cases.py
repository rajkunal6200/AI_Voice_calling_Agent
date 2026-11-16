"""What the agent must do, stated as checkable expectations.

These are not unit tests. A unit test asserts that ``check_stock`` returns the
right row given a SKU; that is deterministic and lives in ``tests/``. These
assert something a test cannot: that the *model* decides to call ``check_stock``
at all, given something a caller might actually say.

They are therefore non-deterministic, cost money to run, and are scored rather
than passed or failed. Run them after changing a prompt, a model, or the tool
split — the three things that silently change behaviour without breaking a test.

Every case here corresponds to a real failure seen during development.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Case:
    id: str
    #: What the caller says, phrased as a person would rather than as a prompt.
    utterance: str
    #: Tool that must be called. Name the tool that *answers* the question,
    #: not the route taken to reach it — otherwise the case only passes in one
    #: agent mode, and the thing being tested is the plumbing rather than
    #: whether the caller got an answer.
    expect_tool: str | None = None
    #: Any one of these will do. For questions more than one tool answers
    #: honestly, where naming a single tool asserts a preference nobody holds
    #: and fails on an answer that was actually correct.
    expect_any_tool: list[str] = field(default_factory=list)
    #: Tools that must NOT be called. The counterpart to expect_tool for cases
    #: where the requirement is restraint: some replies are fine and only one
    #: specific action would be wrong.
    forbid_tools: list[str] = field(default_factory=list)
    #: Substrings that must appear in the spoken reply, lowercased.
    expect_says: list[str] = field(default_factory=list)
    #: Substrings that must NOT appear — usually invented facts.
    forbid_says: list[str] = field(default_factory=list)
    #: If any of these appear in the reply, the named tool must have run first.
    #: Expresses "don't state this from memory" without forbidding it outright.
    grounded_by: tuple[list[str], str] | None = None
    #: Why this case exists, printed on failure.
    because: str = ""


CASES: list[Case] = [
    # --- getting to an answer ----------------------------------------------
    Case(
        id="route.stock",
        utterance="Do you have the desk lamp in stock?",
        expect_tool="check_stock",
        because="reception must route product questions rather than answer them",
    ),
    Case(
        id="route.order",
        utterance="Where's my order?",
        expect_tool="find_my_recent_orders",
        because="an order question with no reference still belongs to orders",
    ),
    Case(
        id="route.policy",
        utterance="What's your returns policy?",
        expect_tool="search_policies",
        because="policy questions must reach the agent with the knowledge base",
    ),
    Case(
        id="route.booking",
        utterance="Can someone come out and assemble it?",
        expect_tool="offer_appointment_slots",
        because="assembly visits are bookings, not sales",
    ),
    # --- grounding: the failure that started all this -----------------------
    Case(
        id="ground.unstocked_product",
        utterance="Do you sell office chairs?",
        # Either tool answers this honestly: check_stock looks the category up,
        # list_products reads back the real catalogue and names the chairs in
        # it. The case originally demanded check_stock because list_products
        # did not exist yet; keeping that demand scored a grounded, correct,
        # arguably better answer as a failure. What is actually being guarded
        # is that *some* tool was consulted, which is what the original bug
        # skipped.
        expect_any_tool=["check_stock", "list_products"],
        forbid_says=["we sell a wide range", "we carry a variety"],
        because=(
            "the original bug: the agent answered 'Yes we do' from general "
            "knowledge of furniture shops, without consulting the catalogue"
        ),
    ),
    Case(
        id="ground.no_invented_price",
        utterance="How much is a filing cabinet?",
        expect_tool="check_stock",
        grounded_by=(["dollars", "$"], "check_stock"),
        because=(
            "a price may be quoted, but only one that came from check_stock — "
            "quoting from memory is how the model invents a plausible figure"
        ),
    ),
    # --- reachability -------------------------------------------------------
    Case(
        id="reach.browse_catalogue",
        utterance="Can you tell me about the furniture you sell?",
        expect_tool="list_products",
        because=(
            "nothing could answer this — check_stock needs a named product — "
            "so sales handed back, reception routed to sales, and the call "
            "ping-ponged until the caller gave up"
        ),
    ),
    Case(
        id="reach.update_own_name",
        utterance="Actually my name is spelled wrong, it's Okonkwo with a K.",
        expect_tool="save_caller_details",
        because=(
            "save_caller_details lived only on SalesAgent, so an agent without "
            "it promised a callback instead — a name change should never leave "
            "the conversation"
        ),
    ),
    Case(
        id="reach.no_needless_callback",
        utterance="Can you update my email address please?",
        forbid_says=["colleague will", "someone will call", "call you back"],
        because=(
            "promising a callback for work the agent can do itself leaves the "
            "caller waiting for a call that was never needed"
        ),
    ),
    # --- buying: interest and commitment are different writes ---------------
    # Two tools sit close together here: one opens an opportunity in the CRM,
    # the other takes stock and creates an order in the ERP. Only the model
    # decides which, and only from how the caller phrased it — so this is
    # exactly the kind of judgement a unit test cannot reach.
    Case(
        id="sales.hesitation",
        utterance="I'm thinking about a desk lamp but I need to measure first.",
        # Stated as restraint rather than as a required call on purpose. The
        # model usually offers to check the price instead of logging an
        # opportunity, and that is a perfectly good thing to say to someone
        # who just told you they are not ready. Only one action is actually
        # wrong here, so only that one is asserted.
        forbid_tools=["place_order"],
        forbid_says=["ord-", "your order is", "i've placed"],
        because=(
            "a maybe must not consume stock; placing an order on hesitation is "
            "a real order the caller never agreed to"
        ),
    ),
    Case(
        id="sales.commit",
        utterance="Yes, go ahead and order two desk lamps for me please.",
        expect_tool="place_order",
        because=(
            "a clear yes must produce a real order, not another opportunity "
            "the caller never hears about again"
        ),
    ),
    # --- escalation ---------------------------------------------------------
    Case(
        id="escalate.out_of_scope",
        utterance="I want to complain about a damaged delivery and speak to a manager.",
        because=(
            "no tool covers complaints; the caller must be routed or escalated, "
            "never told something reassuring and invented"
        ),
        forbid_says=["refund has been", "i have processed", "i've arranged"],
    ),
]
