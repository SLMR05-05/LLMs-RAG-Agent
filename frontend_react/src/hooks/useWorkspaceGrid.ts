import { useMemo } from 'react';

interface UseWorkspaceGridArgs {
    leftCollapsed: boolean;
    rightCollapsed: boolean;
}

export function useWorkspaceGrid({
    leftCollapsed,
    rightCollapsed,
}: UseWorkspaceGridArgs): string {
    return useMemo(() => {
        if (leftCollapsed && rightCollapsed) return '48px 1fr 48px';
        if (leftCollapsed) return '48px 1fr 310px';
        if (rightCollapsed) return '300px 1fr 48px';
        return '300px 1fr 310px';
    }, [leftCollapsed, rightCollapsed]);
}
