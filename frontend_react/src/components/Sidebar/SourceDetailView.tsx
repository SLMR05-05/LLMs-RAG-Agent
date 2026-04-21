import type { FC } from 'react';
import { useEffect, useMemo, useState } from 'react';
import { ArrowLeft, FileText, Image, Loader2 } from 'lucide-react';
import { useAppStore, type SourceDocument } from '../../store/useAppStore';
import { apiService } from '../../services/api';

interface SourceDetailViewProps {
    notebookId: string;
    source: SourceDocument;
}

type HighlightRange = { start: number; end: number } | null;

function getExtension(source: SourceDocument): string {
    const extFromDescription = (source.description || '').replace(/^\./, '').toLowerCase();
    if (extFromDescription) {
        return extFromDescription;
    }

    const fileName = source.title || '';
    return fileName.includes('.') ? fileName.split('.').pop()?.toLowerCase() || '' : '';
}

function buildRawText(source: SourceDocument): string {
    if (source.parsed_markdown?.trim()) {
        return source.parsed_markdown;
    }

    const chunks = source.chunks || [];
    return chunks.map((chunk) => chunk?.text_content || '').filter(Boolean).join('\n\n');
}

function normalizeForSearch(value: string): string {
    return value
        .normalize('NFD')
        .replace(/[\u0300-\u036f]/g, '')
        .toLowerCase()
        .replace(/[^a-z0-9\s]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
}

function findBestFuzzyRange(content: string, snippet: string): HighlightRange {
    const loweredContent = content.toLowerCase();
    const loweredSnippet = snippet.trim().toLowerCase();

    if (!loweredSnippet) {
        return null;
    }

    const exactIndex = loweredContent.indexOf(loweredSnippet);
    if (exactIndex >= 0) {
        return { start: exactIndex, end: exactIndex + loweredSnippet.length };
    }

    const paragraphs = content.split(/\n{2,}/).filter(Boolean);
    if (paragraphs.length === 0) {
        return null;
    }

    const snippetTokens = new Set(normalizeForSearch(snippet).split(' ').filter((token) => token.length >= 3));
    if (snippetTokens.size === 0) {
        return null;
    }

    let cursor = 0;
    let bestScore = 0;
    let bestRange: HighlightRange = null;

    for (const paragraph of paragraphs) {
        const normalizedParagraph = normalizeForSearch(paragraph);
        const paragraphTokens = new Set(normalizedParagraph.split(' ').filter((token) => token.length >= 3));

        let overlap = 0;
        snippetTokens.forEach((token) => {
            if (paragraphTokens.has(token)) {
                overlap += 1;
            }
        });

        const score = overlap / snippetTokens.size;
        if (score > bestScore) {
            const start = content.indexOf(paragraph, cursor);
            const end = start >= 0 ? start + paragraph.length : cursor + paragraph.length;
            bestScore = score;
            bestRange = start >= 0 ? { start, end } : null;
        }

        cursor += paragraph.length + 2;
    }

    return bestScore >= 0.35 ? bestRange : null;
}

function renderHighlightedText(content: string, highlightRange: HighlightRange) {
    if (!highlightRange) {
        return <span>{content}</span>;
    }

    const before = content.slice(0, highlightRange.start);
    const highlighted = content.slice(highlightRange.start, highlightRange.end);
    const after = content.slice(highlightRange.end);

    return (
        <>
            <span>{before}</span>
            <mark id="active-citation-highlight" className="bg-yellow-200 text-black font-bold">
                {highlighted}
            </mark>
            <span>{after}</span>
        </>
    );
}

export const SourceDetailView: FC<SourceDetailViewProps> = ({ notebookId, source }) => {
    const closeSourceDetail = useAppStore((s) => s.closeSourceDetail);
    const activeCitation = useAppStore((s) => s.activeCitation);

    const [detail, setDetail] = useState<SourceDocument | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [loadError, setLoadError] = useState<string | null>(null);

    useEffect(() => {
        let mounted = true;
        setDetail(null);

        const loadSourceDetail = async () => {
            setIsLoading(true);
            setLoadError(null);

            try {
                const data = await apiService.getSourceDetail(notebookId, source.id);
                if (mounted) {
                    setDetail({ ...source, ...data, selected: source.selected });
                }
            } catch {
                if (mounted) {
                    setLoadError('Không thể tải chi tiết tài liệu.');
                }
            } finally {
                if (mounted) {
                    setIsLoading(false);
                }
            }
        };

        void loadSourceDetail();

        return () => {
            mounted = false;
        };
    }, [notebookId, source.id, source.selected, source.title, source.description]);

    const resolvedSource = detail || source;
    const extension = useMemo(() => getExtension(resolvedSource), [resolvedSource]);
    const rawText = useMemo(() => buildRawText(resolvedSource), [resolvedSource]);
    const previewUrl = apiService.getSourcePreviewUrl(notebookId, source.id);

    const isImage = ['png', 'jpg', 'jpeg', 'bmp', 'gif', 'tif', 'tiff', 'webp'].includes(extension);
    const isWebLink = resolvedSource.type === 'web';
    const sourceUrl = resolvedSource.title?.startsWith('http') ? resolvedSource.title : undefined;

    const highlightRange = useMemo(() => {
        if (!activeCitation || !rawText) {
            return null;
        }

        const sameById = Boolean(activeCitation.sourceId) && activeCitation.sourceId === resolvedSource.id;
        const sameByName = normalizeForSearch(activeCitation.sourceName) === normalizeForSearch(resolvedSource.title);
        const closeByName =
            normalizeForSearch(resolvedSource.title).includes(normalizeForSearch(activeCitation.sourceName)) ||
            normalizeForSearch(activeCitation.sourceName).includes(normalizeForSearch(resolvedSource.title));

        if (!sameById && !sameByName && !closeByName) {
            return null;
        }

        return findBestFuzzyRange(rawText, activeCitation.snippet);
    }, [activeCitation, rawText, resolvedSource.id, resolvedSource.title]);

    useEffect(() => {
        if (!highlightRange) {
            return;
        }

        const scrollTimer = window.setTimeout(() => {
            const highlightElement = document.getElementById('active-citation-highlight');
            if (highlightElement) {
                highlightElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }, 80);

        return () => window.clearTimeout(scrollTimer);
    }, [highlightRange]);

    return (
        <div className="flex h-full min-h-0 flex-col animate-fade-in">
            <div className="flex items-center gap-3 border-b border-gray-200 px-4 py-3">
                <button
                    type="button"
                    onClick={closeSourceDetail}
                    className="inline-flex items-center gap-1 rounded-lg border border-gray-200 px-2.5 py-1.5 text-xs font-semibold text-gray-700 transition-colors hover:bg-gray-100"
                >
                    <ArrowLeft className="h-3.5 w-3.5" />
                    Quay lại
                </button>

                <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 text-sm font-semibold text-gray-900">
                        {isImage ? <Image className="h-4 w-4 text-blue-600" /> : <FileText className="h-4 w-4 text-blue-600" />}
                        <span className="truncate">{resolvedSource.title}</span>
                    </div>
                </div>
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto max-h-[calc(100vh-150px)] px-4 py-3">
                {isLoading && (
                    <div className="mb-3 flex items-center gap-2 rounded-xl border border-gray-200 bg-gray-50 px-3 py-2 text-sm text-gray-600">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Đang tải nội dung tài liệu...
                    </div>
                )}

                {loadError && (
                    <div className="mb-3 rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                        {loadError}
                    </div>
                )}

                {!isLoading && !loadError && (
                    <div className="space-y-4">
                        {isImage && (
                            <div className="rounded-2xl border border-gray-200 bg-white p-3">
                                <img
                                    src={previewUrl}
                                    alt={resolvedSource.title}
                                    className="max-h-64 w-full rounded-xl object-contain bg-gray-50"
                                    loading="lazy"
                                />
                            </div>
                        )}

                        <div className="rounded-2xl border border-gray-200 bg-white p-3">
                            {isWebLink && sourceUrl && (
                                <a
                                    href={sourceUrl}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="mb-3 inline-flex text-xs font-medium text-blue-600 underline underline-offset-2 hover:text-blue-700"
                                >
                                    Mở trang gốc: {sourceUrl}
                                </a>
                            )}

                            <div className="whitespace-pre-wrap text-sm leading-7 text-gray-800">
                                {rawText ? renderHighlightedText(rawText, highlightRange) : 'Không có nội dung để hiển thị.'}
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default SourceDetailView;
