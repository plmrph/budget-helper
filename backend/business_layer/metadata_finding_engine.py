"""
MetadataFindingEngine implementation.

This Engine performs atomic metadata finding activities including searching
for email receipts and other metadata sources related to transactions.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from configs import ConfigKeys, FieldNames
from thrift_gen.databasestoreaccess.ttypes import (
    Filter,
    FilterOperator,
    FilterValue,
    Query,
)
from thrift_gen.entities.ttypes import (
    EntityType,
    MetadataType,
    Transactions,
)
from thrift_gen.exceptions.ttypes import (
    InternalException,
    NotFoundException,
    RemoteServiceException,
    UnauthorizedException,
    ValidationException,
)
from thrift_gen.metadatafindingengine.ttypes import MetadataCandidate, MetadataFilter
from thrift_gen.metadatasourceaccess.ttypes import (
    MetadataSourceQuery,
    MetadataSourceType,
)
from thrift_gen.entities.ttypes import PrimitiveValue

logger = logging.getLogger(__name__)


class MetadataFindingEngine:
    """
    MetadataFindingEngine implements atomic metadata finding activities.

    This Engine searches for and identifies metadata (emails, receipts, etc.)
    that may be related to transactions.

    Public methods correspond exactly to the MetadataFindingEngine Thrift interface.
    All other methods are private (prefixed with _).
    """

    def __init__(
        self,
        metadata_source_access=None,
        database_store_access=None,
        config_service=None,
    ):
        """
        Initialize MetadataFindingEngine.

        Args:
            metadata_source_access: MetadataSourceAccess instance
            database_store_access: DatabaseStoreAccess instance
            config_service: ConfigService instance
        """
        self.metadata_source_access = metadata_source_access
        self.database_store_access = database_store_access
        self.config_service = config_service

        logger.info("MetadataFindingEngine initialized")

    async def getMetadataCandidates(
        self, transactions: Transactions, filter: MetadataFilter | None = None
    ) -> list[MetadataCandidate]:
        """
        Find metadata candidates for transactions using enhanced matching algorithms.

        Args:
            transactions: Transactions to find metadata for
            filter: Optional filter for metadata sources

        Returns:
            List of metadata candidates sorted by relevance

        Raises:
            NotFoundException: If no metadata found
            ValidationException: If validation fails
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
            RemoteServiceException: If remote service error occurs
        """
        try:
            logger.info("Finding metadata candidates for transactions")

            # Enhanced input validation
            if not transactions:
                raise ValidationException("Transactions parameter is required")

            if not transactions.transactions and not transactions.transactionIds:
                raise ValidationException(
                    "Either transactions list or transaction IDs must be provided"
                )

            # Validate filter if provided
            if filter:
                self._validate_metadata_filter(filter)

            candidates = []

            # Process transactions based on type
            transaction_list = []
            if transactions.transactions:
                transaction_list = transactions.transactions
                logger.info(f"Processing {len(transaction_list)} transactions directly")
            elif transactions.transactionIds:
                # Fetch transactions by IDs
                logger.info(
                    f"Fetching {len(transactions.transactionIds)} transactions by ID"
                )
                transaction_list = await self._fetch_transactions_by_ids(
                    transactions.transactionIds
                )

            if not transaction_list:
                raise ValidationException("No valid transactions found to process")

            # Find metadata for each transaction with enhanced error handling
            processed_count = 0
            error_count = 0

            import time

            start_time = time.time()

            async def process_transaction(transaction):
                try:
                    # Validate individual transaction
                    if not self._validate_transaction(transaction):
                        logger.warning(
                            f"Skipping invalid transaction: {getattr(transaction, 'id', 'unknown')}"
                        )
                        return []

                    return await self._find_metadata_for_transaction(
                        transaction, filter
                    )
                except Exception as e:
                    transaction_id = getattr(transaction, "id", "unknown")
                    logger.warning(
                        f"Error finding metadata for transaction {transaction_id}: {e}"
                    )
                    return []

            # Process transactions concurrently
            results = await asyncio.gather(
                *(process_transaction(transaction) for transaction in transaction_list),
                return_exceptions=True,
            )

            # Collect results and handle errors
            for result in results:
                if isinstance(result, list):
                    candidates.extend(result)
                    processed_count += 1
                else:
                    error_count += 1

            if error_count > len(transaction_list) * 0.5:  # More than 50% errors
                logger.error("Too many errors processing transactions, stopping")
                raise InternalException(
                    "Failed to process transactions due to excessive errors"
                )

            end_time = time.time()
            logger.info(
                f"Processed {processed_count} transactions in {end_time - start_time:.2f} seconds, found {len(candidates)} metadata candidates"
            )

            logger.info(
                f"Processed {processed_count} transactions, found {len(candidates)} metadata candidates"
            )

            # Apply global filtering and deduplication
            filtered_candidates = self._apply_global_filters(candidates, filter)

            return filtered_candidates

        except (
            ValidationException,
            NotFoundException,
            UnauthorizedException,
            RemoteServiceException,
        ):
            raise
        except Exception as e:
            logger.error(f"Error finding metadata candidates: {e}")
            raise InternalException(
                f"Failed to find metadata candidates: {str(e)}"
            ) from e

    def _validate_metadata_filter(self, filter: MetadataFilter) -> None:
        """
        Validate metadata filter parameters.

        Args:
            filter: Metadata filter to validate

        Raises:
            ValidationException: If filter is invalid
        """
        if filter.startTime and filter.endTime:
            if filter.startTime >= filter.endTime:
                raise ValidationException("Filter start time must be before end time")

        if filter.sourceTypes:
            valid_types = [MetadataType.Email, MetadataType.Prediction]
            for source_type in filter.sourceTypes:
                if source_type not in valid_types:
                    raise ValidationException(f"Invalid source type: {source_type}")

    def _validate_transaction(self, transaction) -> bool:
        """
        Validate individual transaction for metadata finding.

        Args:
            transaction: Transaction to validate

        Returns:
            True if transaction is valid for processing
        """
        try:
            # Must have at least one searchable field
            has_payee = hasattr(transaction, "payeeId") and bool(transaction.payeeId)
            has_memo = hasattr(transaction, "memo") and bool(transaction.memo)
            has_amount = (
                hasattr(transaction, "amount") and transaction.amount is not None
            )

            return has_payee or has_memo or has_amount

        except Exception:
            return False

    def _apply_global_filters(
        self, candidates: list[MetadataCandidate], filter: MetadataFilter | None
    ) -> list[MetadataCandidate]:
        """
        Apply global filtering and deduplication to candidates.

        Args:
            candidates: List of metadata candidates
            filter: Optional metadata filter

        Returns:
            Filtered and deduplicated candidates
        """
        try:
            if not candidates:
                return candidates

            # Remove duplicates based on metadata content
            unique_candidates = self._deduplicate_candidates(candidates)

            # Apply search phrase filter if specified
            if filter and filter.searchPhrase:
                phrase_lower = filter.searchPhrase.lower()
                filtered_candidates = []

                for candidate in unique_candidates:
                    if (
                        candidate.metadata.value
                        and candidate.metadata.value.stringValue
                        and phrase_lower in candidate.metadata.value.stringValue.lower()
                    ):
                        filtered_candidates.append(candidate)

                unique_candidates = filtered_candidates

            # Sort by relevance (already scored in _score_metadata_candidates)
            # Candidates are already sorted, but ensure consistency

            logger.info(
                f"Applied global filters: {len(candidates)} -> {len(unique_candidates)} candidates"
            )
            return unique_candidates

        except Exception as e:
            logger.warning(f"Error applying global filters: {e}")
            return candidates

    def _deduplicate_candidates(
        self, candidates: list[MetadataCandidate]
    ) -> list[MetadataCandidate]:
        """
        Remove duplicate candidates based on content similarity.

        Args:
            candidates: List of metadata candidates

        Returns:
            Deduplicated candidates
        """
        try:
            if not candidates:
                return candidates

            unique_candidates = []
            seen_content = set()

            for candidate in candidates:
                if not candidate.metadata or not candidate.metadata.value:
                    continue

                content = candidate.metadata.value.stringValue
                if not content:
                    continue

                # Create a normalized key for deduplication
                content_key = content.lower().strip()[
                    :100
                ]  # First 100 chars, normalized

                if content_key not in seen_content:
                    seen_content.add(content_key)
                    unique_candidates.append(candidate)

            return unique_candidates

        except Exception as e:
            logger.warning(f"Error deduplicating candidates: {e}")
            return candidates

    async def _fetch_transactions_by_ids(self, transaction_ids: list[str]) -> list:
        """
        Fetch transactions by their IDs.

        Args:
            transaction_ids: List of transaction IDs

        Returns:
            List of transaction objects
        """
        try:
            transactions = []

            for transaction_id in transaction_ids:
                filter_value = FilterValue(stringValue=transaction_id)
                id_filter = Filter(
                    fieldName=FieldNames.ID,
                    operator=FilterOperator.EQ,
                    value=filter_value,
                )

                query = Query(
                    entityType=EntityType.Transaction, filters=[id_filter], limit=1
                )

                result = await self.database_store_access.getEntities(query)

                if result.entities and result.entities[0].transaction:
                    transactions.append(result.entities[0].transaction)

            return transactions

        except Exception as e:
            logger.error(f"Error fetching transactions by IDs: {e}")
            return []

    async def _find_metadata_for_transaction(
        self, transaction, filter: MetadataFilter | None = None
    ) -> list[MetadataCandidate]:
        """
        Find metadata candidates for a single transaction.

        This method:
        1. Checks local database for existing metadata
        2. Calls out to external metadata sources (Gmail, etc.) to find new candidates

        Args:
            transaction: Transaction to find metadata for
            filter: Optional source filter

        Returns:
            List of metadata candidates for this transaction
        """
        try:
            candidates = []

            # 1. First check local database for existing metadata attached to this transaction
            local_candidates = await self._get_local_metadata_candidates(
                transaction, filter
            )
            candidates.extend(local_candidates)

            # 2. Search external metadata sources for new candidates
            external_candidates = await self._search_external_metadata_sources(
                transaction, filter
            )
            candidates.extend(external_candidates)

            return candidates

        except Exception as e:
            logger.error(
                f"Error finding metadata for transaction {transaction.id}: {e}"
            )
            return []

    async def _get_local_metadata_candidates(
        self, transaction, filter: MetadataFilter | None = None
    ) -> list[MetadataCandidate]:
        """
        Get metadata candidates from local database for this transaction.

        Since metadata is stored directly in the transaction entity, we get it from there
        rather than querying separately.

        Args:
            transaction: Transaction to get local metadata for
            filter: Optional source filter

        Returns:
            List of local metadata candidates
        """
        try:
            candidates = []

            # Get metadata directly from the transaction entity
            if transaction.metadata:
                for metadata in transaction.metadata:
                    # Apply filter if specified
                    if filter and filter.sourceTypes:
                        if metadata.type not in filter.sourceTypes:
                            continue

                    candidate = MetadataCandidate(metadata=metadata)
                    candidates.append(candidate)

            logger.info(
                f"Found {len(candidates)} local metadata candidates for transaction {transaction.id}"
            )
            return candidates

        except Exception as e:
            logger.error(f"Error getting local metadata candidates: {e}")
            return []

    async def _search_external_metadata_sources(
        self, transaction, filter: MetadataFilter | None = None
    ) -> list[MetadataCandidate]:
        """
        Search external metadata sources for new candidates.

        Args:
            transaction: Transaction to search metadata for
            filter: Optional source filter

        Returns:
            List of external metadata candidates
        """
        try:
            candidates = []

            # Determine search parameters - use custom search phrase if provided
            if filter and filter.searchPhrase:
                search_terms = [filter.searchPhrase]
                logger.info(f"Using custom search phrase: {filter.searchPhrase}")
            else:
                search_terms = await self._extract_search_terms(transaction)
                logger.info(f"Using automatic search terms: {search_terms}")

            if not search_terms:
                logger.info(f"No search terms found for transaction {transaction.id}")
                return candidates

            # Search email sources if enabled
            if (
                not filter
                or not filter.sourceTypes
                or MetadataType.Email in filter.sourceTypes
            ):
                email_candidates = await self._search_email_metadata(
                    transaction, search_terms, filter
                )
                candidates.extend(email_candidates)

            # Search prediction sources if enabled
            if (
                not filter
                or not filter.sourceTypes
                or MetadataType.Prediction in filter.sourceTypes
            ):
                prediction_candidates = await self._search_prediction_metadata(
                    transaction, filter
                )
                candidates.extend(prediction_candidates)

            # Future: Add other metadata source types here
            # - SMS receipts
            # - Photo receipts

            logger.info(
                f"Found {len(candidates)} external metadata candidates for transaction {transaction.id}"
            )
            return candidates

        except Exception as e:
            logger.error(f"Error searching external metadata sources: {e}")
            return []

    async def _search_prediction_metadata(
        self, transaction, filter: MetadataFilter | None = None
    ) -> list[MetadataCandidate]:
        """
        Search for prediction metadata (placeholder for now).

        Args:
            transaction: Transaction to get predictions for
            filter: Optional source filter

        Returns:
            List of prediction metadata candidates
        """
        try:
            # Placeholder - prediction metadata not implemented yet
            logger.info(
                f"Prediction metadata search not implemented yet for transaction {transaction.id}"
            )
            return []

        except Exception as e:
            logger.error(f"Error searching prediction metadata: {e}")
            return []

    async def _extract_search_terms(self, transaction) -> list[str]:
        """
        Extract search terms from transaction data using enhanced matching patterns.

        Args:
            transaction: Transaction to extract terms from

        Returns:
            List of search terms prioritized by relevance
        """
        terms = []

        payee = None
        amount_or_term = None

        # Prepare amount variations
        if hasattr(transaction, "amount") and transaction.amount:
            amount_dollars = abs(transaction.amount) / 1000.0
            whole = int(amount_dollars)
            decimal_str = f"{amount_dollars:.2f}"
            whole_str = str(whole)
            whole_comma = f"{whole:,}"
            decimal_comma = f"{float(decimal_str):,.2f}"

            # Always include precise decimal formats
            variations = [
                f"${decimal_comma}",  # e.g., $30.67
                f"{decimal_str}",  # e.g., 30.67
            ]

            # Only include whole-dollar variants when amount has zero cents
            if amount_dollars.is_integer():
                variations.insert(0, f"${whole_comma}")  # e.g., $1,234
                variations.append(f"${whole_str}")  # e.g., $30

            # Deduplicate while preserving order and quote for phrase search
            variations = [f'"{v}"' for v in dict.fromkeys(variations)]
            amount_or_term = " OR ".join(variations)

        # Prepare payee variations
        payee_variations = []
        # Try to get payee name from transaction, or fetch from DB if only payeeId is present
        payee_name = None
        if hasattr(transaction, "payeeName") and transaction.payeeName:
            payee_name = transaction.payeeName.strip()
        elif hasattr(transaction, "payeeId") and transaction.payeeId:
            payee_id = transaction.payeeId.strip()
            try:
                # Query for specific payee by ID instead of fetching all payees
                payee_entities = await self.database_store_access.getEntitiesById(
                    EntityType.Payee, [payee_id]
                )
                if payee_entities and payee_entities[0].payee:
                    payee_name = payee_entities[0].payee.name
                else:
                    logger.warning(f"Payee not found for ID: {payee_id}")
                    payee_name = None
            except Exception as e:
                logger.error(f"Error fetching payee name for ID {payee_id}: {e}")
                payee_name = None

        if payee_name:
            payee = payee_name
            payee_variations.append(payee)

            if len(payee) > 4:
                first_word = payee.split()[0] if " " in payee else payee
                if len(first_word) > 3:
                    payee_variations.append(first_word)

                cleaned_payee = (
                    payee.replace(" LLC", "").replace(" Inc", "").replace(" Corp", "")
                )
                if cleaned_payee != payee:
                    payee_variations.append(cleaned_payee)

        # 1. Highest priority: payee_name AND (amount terms OR'd)
        if len(payee_variations) > 1 and amount_or_term:
            terms.append(f"{payee_variations[1]} ({amount_or_term})")

        # 2. Next: just amount terms OR'd
        if amount_or_term:
            terms.append(amount_or_term)

        # 3. Next: just payee variations
        for v in payee_variations:
            terms.append(v)

        # Remove duplicates while preserving order (priority)
        seen = set()
        unique_terms = []
        for term in terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        return unique_terms

    async def _search_email_metadata(
        self,
        transaction,
        search_terms: list[str],
        filter: MetadataFilter | None = None,
    ) -> list[MetadataCandidate]:
        """
        Search for email metadata related to transaction.

        Args:
            transaction: Transaction to search for
            search_terms: Search terms extracted from transaction
            filter: Optional source filter

        Returns:
            List of email metadata candidates
        """
        try:
            candidates = []

            # Build search queries for metadata source access

            # Determine search time window
            transaction_date = self._parse_transaction_date(transaction)
            start_time, end_time = await self._get_search_time_window(
                transaction_date, filter
            )

            # Create search queries for each term
            queries = []
            for term in search_terms[
                :2
            ]:  # Limit to top 2 terms to avoid too many queries
                query = MetadataSourceQuery(
                    sourceType=MetadataSourceType.Email,
                    searchPhrase=term,
                    startTime=start_time,
                    endTime=end_time,
                    limit=10,  # Default limit for email search
                )
                queries.append(query)

            if not queries:
                return candidates

            # Search for metadata
            metadata_results = await self.metadata_source_access.getMetadata(queries)

            # Convert results to candidates
            for metadata in metadata_results:
                if metadata.type == MetadataType.Email:
                    candidate = MetadataCandidate(metadata=metadata)
                    candidates.append(candidate)

            # Score and rank candidates by relevance
            scored_candidates = self._score_metadata_candidates(
                candidates, transaction, search_terms
            )
            return scored_candidates

        except Exception as e:
            logger.error(f"Error searching email metadata: {e}")
            return []

    def _parse_transaction_date(self, transaction) -> datetime:
        """
        Parse transaction date.

        Args:
            transaction: Transaction object

        Returns:
            Transaction date as datetime
        """
        try:
            if hasattr(transaction, "date") and transaction.date:
                if isinstance(transaction.date, str):
                    return datetime.fromisoformat(
                        transaction.date.replace("Z", "+00:00")
                    )
                elif isinstance(transaction.date, datetime):
                    return transaction.date

            # Default to current time if no date available
            return datetime.utcnow()

        except Exception as e:
            logger.warning(f"Error parsing transaction date: {e}")
            return datetime.utcnow()

    async def _get_search_time_window(
        self, transaction_date: datetime, filter: MetadataFilter | None = None
    ) -> tuple:
        """
        Get search time window around transaction date.

        Args:
            transaction_date: Transaction date
            filter: Optional source filter with time constraints

        Returns:
            Tuple of (start_time, end_time) as timestamps
        """
        search_days_buffer = await self.config_service.getConfigValue(
            ConfigKeys.METADATA_SEARCH_DAYS_BUFFER
        )
        # Fallback to default if config value is None
        if search_days_buffer is None:
            search_days_buffer = 3

        default_start = transaction_date - timedelta(days=search_days_buffer)
        default_end = transaction_date + timedelta(days=search_days_buffer)

        # Apply filter constraints if provided
        if filter:
            if filter.startTime:
                start_time = max(
                    default_start, datetime.fromtimestamp(filter.startTime)
                )
            else:
                start_time = default_start

            if filter.endTime:
                end_time = min(default_end, datetime.fromtimestamp(filter.endTime))
            else:
                end_time = default_end
        else:
            start_time = default_start
            end_time = default_end

        # Convert to timestamps
        return int(start_time.timestamp()), int(end_time.timestamp())

    def _score_metadata_candidates(
        self, candidates: list[MetadataCandidate], transaction, search_terms: list[str]
    ) -> list[MetadataCandidate]:
        """
        Score and rank metadata candidates by relevance.

        Args:
            candidates: List of metadata candidates
            transaction: Transaction being matched
            search_terms: Search terms used

        Returns:
            Sorted list of candidates by relevance score
        """
        try:
            scored_candidates = []

            for candidate in candidates:
                score = self._calculate_relevance_score(
                    candidate, transaction, search_terms
                )
                # Store numeric relevance in metadata.properties as a PrimitiveValue.doubleValue
                if candidate.metadata.properties is None:
                    candidate.metadata.properties = {}
                # Use key 'relevance' to store the numeric score
                candidate.metadata.properties["relevance"] = PrimitiveValue(doubleValue=score)

                scored_candidates.append((score, candidate))

            # Sort by score (highest first) and return candidates
            scored_candidates.sort(key=lambda x: x[0], reverse=True)

            return [candidate for score, candidate in scored_candidates]

        except Exception as e:
            logger.warning(f"Error scoring metadata candidates: {e}")
            return candidates

    def _calculate_relevance_score(
        self, candidate: MetadataCandidate, transaction, search_terms: list[str]
    ) -> float:
        """
        Calculate relevance score for a metadata candidate using enhanced matching algorithms.

        Args:
            candidate: Metadata candidate
            transaction: Transaction being matched
            search_terms: Search terms used

        Returns:
            Relevance score (0.0 to 1.0)
        """
        try:
            score = 0.0

            if not candidate.metadata or not candidate.metadata.value:
                return score

            # Get metadata content for scoring
            content = ""
            if candidate.metadata.value.stringValue:
                content = candidate.metadata.value.stringValue.lower()

            if not content:
                return score

            # Enhanced term matching with weighted scoring
            weighted_matches = 0.0
            total_weight = 0.0

            for i, term in enumerate(search_terms):
                # Higher weight for earlier terms (they're more important)
                weight = 1.0 / (i + 1)
                total_weight += weight

                term_lower = term.lower()
                if term_lower in content:
                    # Exact match gets full weight
                    weighted_matches += weight
                elif len(term) > 4:
                    # Partial match for longer terms
                    if any(word for word in content.split() if term_lower in word):
                        weighted_matches += weight * 0.7

            if total_weight > 0:
                score += (weighted_matches / total_weight) * 0.5

            # Enhanced amount matching with multiple formats
            if hasattr(transaction, "amount") and transaction.amount:
                amount_dollars = abs(transaction.amount) / 1000.0
                amount_bonus = 0.0

                # Check various amount formats
                amount_formats = [
                    f"{amount_dollars:.2f}",
                    f"${amount_dollars:.2f}",
                    f"{round(amount_dollars):.0f}",
                    f"${round(amount_dollars):.0f}",
                ]

                for amount_format in amount_formats:
                    if amount_format in content:
                        amount_bonus = 0.3
                        break
                    # Fuzzy amount matching (within $1)
                    elif self._fuzzy_amount_match(content, amount_dollars):
                        amount_bonus = 0.2
                        break

                score += amount_bonus

            # Enhanced payee matching
            if hasattr(transaction, "payeeId") and transaction.payeeId:
                payee_bonus = 0.0
                payee_lower = transaction.payeeId.lower()

                if payee_lower in content:
                    payee_bonus = 0.15
                elif len(payee_lower) > 4:
                    # Check if first word of payee is in content
                    first_word = (
                        payee_lower.split()[0] if " " in payee_lower else payee_lower
                    )
                    if first_word in content:
                        payee_bonus = 0.1

                score += payee_bonus

            # Date proximity bonus (if metadata has timestamp)
            if (
                hasattr(candidate.metadata, "timestamp")
                and candidate.metadata.timestamp
            ):
                date_bonus = self._calculate_date_proximity_bonus(
                    candidate.metadata.timestamp,
                    self._parse_transaction_date(transaction),
                )
                score += date_bonus

            return min(score, 1.0)  # Cap at 1.0

        except Exception as e:
            logger.warning(f"Error calculating relevance score: {e}")
            return 0.0

    def _fuzzy_amount_match(self, content: str, amount: float) -> bool:
        """
        Check for fuzzy amount matches within content.

        Args:
            content: Content to search in
            amount: Amount to search for

        Returns:
            True if fuzzy match found
        """
        try:
            import re

            # Find all dollar amounts in content
            dollar_pattern = r"\$?(\d+\.?\d*)"
            matches = re.findall(dollar_pattern, content)

            for match in matches:
                try:
                    found_amount = float(match)
                    # Consider it a match if within $1
                    if abs(found_amount - amount) <= 1.0:
                        return True
                except ValueError:
                    continue

            return False

        except Exception:
            return False

    def _calculate_date_proximity_bonus(
        self, metadata_timestamp: int, transaction_date: datetime
    ) -> float:
        """
        Calculate bonus score based on date proximity between metadata and transaction.

        Args:
            metadata_timestamp: Metadata timestamp
            transaction_date: Transaction date

        Returns:
            Bonus score (0.0 to 0.1)
        """
        try:
            metadata_date = datetime.fromtimestamp(metadata_timestamp)
            time_diff = abs((metadata_date - transaction_date).days)

            # Maximum bonus for same day, decreasing bonus for further dates
            if time_diff == 0:
                return 0.1
            elif time_diff <= 1:
                return 0.08
            elif time_diff <= 3:
                return 0.05
            elif time_diff <= 7:
                return 0.02
            else:
                return 0.0

        except Exception:
            return 0.0
