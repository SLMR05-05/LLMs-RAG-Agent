import type { FC } from 'react';
import { useAppStore } from '../../store/useAppStore';
import { useWorkspaceGrid } from '../../hooks/useWorkspaceGrid';
import LeftSidebar from '../Sidebar/LeftSidebar';
import ChatArea from '../Chat/ChatArea';
import RightStudio from '../Studio/RightStudio';

/**
 * MainLayout - 3-Column Workspace Layout
 * 
 * Responsive CSS Grid layout:
 * - Full: 300px | 1fr | 310px
 * - Left Collapsed: 48px | 1fr | 310px
 * - Right Collapsed: 300px | 1fr | 48px
 * - Both Collapsed: 48px | 1fr | 48px
 */
export const MainLayout: FC = () => {
    const leftCollapsed = useAppStore((s) => s.leftPanelCollapsed);
    const rightCollapsed = useAppStore((s) => s.rightPanelCollapsed);
    const gridTemplateColumns = useWorkspaceGrid({ leftCollapsed, rightCollapsed });

    return (
        <div
            className="h-[calc(100vh-56px)] mt-14 overflow-hidden min-h-0"
            style={{
                display: 'grid',
                gridTemplateColumns,
                gridTemplateRows: '1fr',
                gap: 0,
                transition: 'grid-template-columns 0.28s cubic-bezier(0.2, 0.8, 0.2, 1)',
            }}
        >
            {/* LEFT SIDEBAR: SOURCES */}
            <LeftSidebar collapsed={leftCollapsed} />

            {/* CENTER: CHAT AREA */}
            <ChatArea />

            {/* RIGHT SIDEBAR: STUDIO */}
            <RightStudio collapsed={rightCollapsed} />
        </div>
    );
};

export default MainLayout;
